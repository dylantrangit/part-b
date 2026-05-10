"""I5 MCTS backend.

Builds on the I4 vanilla UCT skeleton (`mcts_uct.py`) and adds:
  - Heavy rollouts via `agent.core.policy.rollout_policy_action`.
  - Early termination at depth 25, returning `tanh(eval/scale)` instead of a
    truncated draw.
  - RAVE / AMAF: per-child `(rave_visits, rave_value)` for the move into that
    child, updated for every action played later in the simulation by the
    parent's side to move.
  - First-play urgency (FPU): unvisited children are scored at
    `parent_q - 0.25` rather than +inf.

Reward convention (same as I4): a leaf's reward is in the perspective of the
player who just moved INTO the leaf. Each node's `total_value` therefore stores
from "moved-into-self" perspective; `backprop` add-then-flips. RAVE rewards on
a child are stored in the same frame so they combine with Q without sign games.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from referee.game import Action

from ..core.eval import evaluate
from ..core.policy import (
    _SOFTMAX_TEMP as _TAU_ROLLOUT,
    _TAU_PRIOR,
    heuristic_score,
    rollout_policy_action,
)
from ..core.time_budget import TimeBudget


from ..search.pvs import _pvs
from ..core.tt import TranspositionTable



_RAVE_BIAS = 1e-5
_FPU_REDUCTION = 0.25
_EVAL_SCALE = 500.0
# Shorter rollouts than the original 25 so we (a) do more iterations per
# unit time and (b) lean more on the calibrated `evaluate()` cutoff than on
# the noisy rollout heuristic. Plan §9.1 picked 25 conservatively; 15 keeps
# enough lookahead for tactical sequences without compounding heuristic noise.
_ROLLOUT_DEPTH_CAP = 12
# PUCT exploration constant. Plan §10.1(b) suggests 1.5–2.5; 2.0 is the
# AlphaZero default and gives a sensible balance between Q-exploitation and
# prior-guided exploration in the 30–80 branching factor of Cascade midgame.
_C_PUCT = 2.0

_HYBRID_MIN_PARENT_VISITS =32
_HYBRID_DEPTH = 5
_HYBRID_BUDGET_SECONDS = 0.01
#comment ts later


@dataclass
class Node:
    parent: "Node | None"
    incoming_move: Action | None
    prior: float = 0.0
    children: dict[Action, "Node"] = field(default_factory=dict)
    untried: list[Action] | None = None
    untried_priors: list[float] | None = None
    untried_rollout_weights: list[float] | None = None
    visits: int = 0
    total_value: float = 0.0
    rave_visits: int = 0
    rave_value: float = 0.0
    terminal_value: float | None = None
    board_hash: int | None = None

def _moved_in_color_name(board) -> str:
    return "BLUE" if board.turn_color.name == "RED" else "RED"

def _try_prove(node: Node) -> None:
    if node.terminal_value is not None:
        return
    if node.untried is None or node.untried:
        return
    if not node.children:
        return

    vals = [child.terminal_value for child in node.children.values()]
    if any(v is None for v in vals):
        return

    if any(v == 1.0 for v in vals):
        node.terminal_value = 1.0
    elif all(v == -1.0 for v in vals):
        node.terminal_value = -1.0
    else:
        node.terminal_value = 0.0


def _terminal_value_for(board, ply: int, perspective: str) -> float | None:
    t = board.terminal(ply)
    if t is None:
        return None
    if t == 0:
        return 0.0
    red_won = t > 0
    if perspective == "RED":
        return 1.0 if red_won else -1.0
    return -1.0 if red_won else 1.0


def _eval_value_for(board, perspective: str) -> float:
    raw = evaluate(board)
    if perspective == "BLUE":
        raw = -raw
    return math.tanh(raw / _EVAL_SCALE)




def _hybrid_value(board) -> float:
    budget = TimeBudget(_HYBRID_BUDGET_SECONDS)
    tt = TranspositionTable()

    try:
        raw, _ = _pvs(
            board,
            _HYBRID_DEPTH,
            -math.inf,
            math.inf,
            0,
            budget,
            tt,
            {},
            {},
        )
    except Exception:
        return _eval_value_for(board, _moved_in_color_name(board))

    # _pvs returns value from side-to-move perspective.
    # child.total_value is stored from moved-into-child perspective,
    # so flip the sign.
    return math.tanh((raw) / _EVAL_SCALE)


def _selection_score(child: Node, parent: Node, c_puct: float) -> float:
    """PUCT score: Q(child) + c_puct · P(child) · √N_parent / (1 + n_child).

    Replaces UCB1 for I5: the heuristic prior breaks UCB1's "every child
    gets explored uniformly first" pattern, which is what was burning all
    the budget on plausibly-terrible openings (cf. the F1–F4 diagnostic).
    """
    n = child.visits
    if n == 0:
        # FPU for the Q term — same logic as before. parent.total_value is
        # in "moved-into-parent" frame, opposite of the parent's side to
        # move; negate so we compare in the same frame as visited siblings.
        if parent.visits > 0:
            q = -parent.total_value / parent.visits - _FPU_REDUCTION
        else:
            q = 0.0
        explore = c_puct * child.prior * math.sqrt(max(parent.visits, 1))
        return q + explore

    q = child.total_value / n
    if child.rave_visits > 0:
        m = child.rave_visits
        rave_q = child.rave_value / m
        beta = m / (m + n + 4.0 * m * n * _RAVE_BIAS)
        q = (1.0 - beta) * q + beta * rave_q

    explore = c_puct * child.prior * math.sqrt(parent.visits) / (1 + n)
    return q + explore

def _select_child(node: Node, c_puct: float) -> Node | None:
    children = list(node.children.values())
    if not children:
        return None

    winning = [c for c in children if c.terminal_value == 1.0]
    if winning:
        return winning[0]

    candidates = [c for c in children if c.terminal_value != -1.0]
    if not candidates:
        candidates = children

    return max(candidates, key=lambda child: _selection_score(child, node, c_puct))

def select(root: Node, board, c: float):
    node = root
    ply = 0
    sim_actions: list[tuple[str, Action]] = []

    while True:
        terminal_value = _terminal_value_for(board, ply, _moved_in_color_name(board))
        if terminal_value is not None:
            node.terminal_value = terminal_value
            return node, ply, sim_actions

        if node.untried is None:
            _materialise_actions(node, board)

        if node.untried:
            return node, ply, sim_actions

        if not node.children:
            return node, ply, sim_actions

        chosen = _select_child(node, c)
        if chosen is None:
            return node, ply, sim_actions

        sim_actions.append((board.turn_color.name, chosen.incoming_move))
        board.apply(chosen.incoming_move)
        ply += 1
        node = chosen


def _materialise_actions(node: Node, board) -> None:
    """Populate untried + priors (τ_prior) + rollout weights (τ_rollout).

    Computes one heuristic score per legal action, then turns it into two
    aligned lists: sharp priors for PUCT selection (τ_prior) and broader
    weights for rollout sampling (τ_rollout). Sorted ascending by raw
    score so `pop()` returns the highest-prior action first.
    """
    actions = list(board.legal_actions())
    if not actions:
        node.untried = []
        node.untried_priors = []
        node.untried_rollout_weights = []
        return
    scores = [heuristic_score(board, a) for a in actions]
    max_s = max(scores)
    prior_weights = [math.exp((s - max_s) / _TAU_PRIOR) for s in scores]
    total = sum(prior_weights)
    if total <= 0.0:
        priors = [1.0 / len(actions)] * len(actions)
    else:
        priors = [w / total for w in prior_weights]
    rollout_weights = [math.exp((s - max_s) / _TAU_ROLLOUT) for s in scores]
    # Sort ascending by raw score so .pop() (LIFO) returns the highest-
    # prior action first — front-loads expansion onto plausible moves.
    order = sorted(range(len(actions)), key=lambda i: scores[i])
    node.untried = [actions[i] for i in order]
    node.untried_priors = [priors[i] for i in order]
    node.untried_rollout_weights = [rollout_weights[i] for i in order]


def _immediate_winning_move(board, ply: int) -> Action | None:
    mover = board.turn_color.name
    for action in board.legal_actions():
        board.apply(action)
        val = _terminal_value_for(board, ply + 1, mover)
        board.undo()
        if val == 1.0:
            return action
    return None

# def expand(node: Node, board, ply: int, sim_actions: list):
#     terminal_value = _terminal_value_for(board, ply, _moved_in_color_name(board))
#     if terminal_value is not None:
#         node.terminal_value = terminal_value
#         return node, ply, False

#     if node.untried is None:
#         _materialise_actions(node, board)

#     if not node.untried:
#         return node, ply, False

#     # force immediate winning move if one exists
#     win_move = _immediate_winning_move(board, ply)
#     if win_move is not None:
#         if win_move in node.children:
#             sim_actions.append((board.turn_color.name, win_move))
#             board.apply(win_move)
#             child = node.children[win_move]
#             child.terminal_value = 1.0
#             return child, ply + 1, False

#         prior = 0.0

#         for i, action in enumerate(node.untried):
#             if action == win_move:
#                 node.untried.pop(i)

#                 if node.untried_priors is not None:
#                     prior = node.untried_priors.pop(i)

#                 if node.untried_rollout_weights is not None:
#                     node.untried_rollout_weights.pop(i)

#                 break

#         sim_actions.append((board.turn_color.name, win_move))
#         board.apply(win_move)

#         child = Node(
#             parent=node,
#             incoming_move=win_move,
#             prior=prior,
#             board_hash=board.zobrist_hash,
#         )
#         child.terminal_value = 1.0
#         node.children[win_move] = child
#         return child, ply + 1, False
    
#     safe_move = _unique_safe_move(board, ply)
#     if safe_move is not None and safe_move not in node.children:
#         prior = 0.0
#         for i, action in enumerate(node.untried):
#             if action == safe_move:
#                 node.untried.pop(i)
#                 if node.untried_priors is not None:
#                     prior = node.untried_priors.pop(i)
#                 if node.untried_rollout_weights is not None:
#                     node.untried_rollout_weights.pop(i)
#                 break
#         else:
#             safe_move = None

#         if safe_move is not None:
#             sim_actions.append((board.turn_color.name, safe_move))
#             print("ANTI-DECISIVE FORCED", safe_move)
#             board.apply(safe_move)

#             child = Node(
#                 parent=node,
#                 incoming_move=safe_move,
#                 prior=prior,
#                 board_hash=board.zobrist_hash,
#             )
#             node.children[safe_move] = child
#             return child, ply + 1, False

#     move = node.untried.pop()
#     prior = node.untried_priors.pop() if node.untried_priors else 0.0
#     if node.untried_rollout_weights is not None:
#         node.untried_rollout_weights.pop()

#     sim_actions.append((board.turn_color.name, move))
#     board.apply(move)

#     child = Node(
#         parent=node,
#         incoming_move=move,
#         prior=prior,
#         board_hash=board.zobrist_hash,
#     )

#     used_hybrid = False
#     # keep hybrid off for now (regressive atp)
#     # if node.visits >= _HYBRID_MIN_PARENT_VISITS: 
#     #     value = _hybrid_value(board)
#     #     child.visits = 1
#     #     child.total_value = value
#     #     used_hybrid = True

#     node.children[move] = child
#     return child, ply + 1, used_hybrid

def expand(node: Node, board, ply: int, sim_actions: list):
    terminal_value = _terminal_value_for(board, ply, _moved_in_color_name(board))
    if terminal_value is not None:
        node.terminal_value = terminal_value
        return node, ply, False

    if node.untried is None:
        _materialise_actions(node, board)

    if not node.untried:
        return node, ply, False

    move = node.untried.pop()
    prior = node.untried_priors.pop() if node.untried_priors else 0.0
    if node.untried_rollout_weights is not None:
        node.untried_rollout_weights.pop()

    sim_actions.append((board.turn_color.name, move))
    board.apply(move)

    child = Node(
        parent=node,
        incoming_move=move,
        prior=prior,
        board_hash=board.zobrist_hash,
    )

    used_hybrid = False
    # keep hybrid off for now
    # if node.visits >= _HYBRID_MIN_PARENT_VISITS:
    #     value = _hybrid_value(board)
    #     child.visits = 1
    #     child.total_value = value
    #     used_hybrid = True

    node.children[move] = child
    return child, ply + 1, used_hybrid


# def expand(node: Node, board, ply: int, sim_actions: list):
#     terminal_value = _terminal_value_for(board, ply, _moved_in_color_name(board))
#     if terminal_value is not None:
#         node.terminal_value = terminal_value
#         return node, ply, False

#     if node.untried is None:
#         _materialise_actions(node, board)

#     if not node.untried:
#         return node, ply, False

#     # force immediate winning move if one exists
#     win_move = _immediate_winning_move(board, ply)
#     if win_move is not None:
#         if win_move in node.children:
#             sim_actions.append((board.turn_color.name, win_move))
#             board.apply(win_move)
#             child = node.children[win_move]
#             child.terminal_value = 1.0
#             return child, ply + 1, False

#         prior = 0.0

#         for i, action in enumerate(node.untried):
#             if action == win_move:
#                 node.untried.pop(i)

#                 if node.untried_priors is not None:
#                     prior = node.untried_priors.pop(i)

#                 if node.untried_rollout_weights is not None:
#                     node.untried_rollout_weights.pop(i)

#                 break

#         sim_actions.append((board.turn_color.name, win_move))
#         board.apply(win_move)

#         child = Node(
#             parent=node,
#             incoming_move=win_move,
#             prior=prior,
#             board_hash=board.zobrist_hash,
#         )
#         child.terminal_value = 1.0
#         node.children[win_move] = child
#         return child, ply + 1, False

#     move = node.untried.pop()
#     prior = node.untried_priors.pop() if node.untried_priors else 0.0
#     if node.untried_rollout_weights is not None:
#         node.untried_rollout_weights.pop()

#     sim_actions.append((board.turn_color.name, move))
#     board.apply(move)

#     child = Node(
#         parent=node,
#         incoming_move=move,
#         prior=prior,
#         board_hash=board.zobrist_hash,
#     )

#     used_hybrid = False
#     # keep hybrid off for now (regressive atm)
#     # if node.visits >= _HYBRID_MIN_PARENT_VISITS:
#     #     value = _hybrid_value(board)
#     #     child.visits = 1
#     #     child.total_value = value
#     #     used_hybrid = True

#     node.children[move] = child
#     return child, ply + 1, used_hybrid




def heavy_rollout(
    board,
    ply: int,
    sim_actions: list,
    depth_cap: int = _ROLLOUT_DEPTH_CAP,
    leaf_actions=None,
    leaf_priors=None,
) -> tuple[float, int]:
    perspective = _moved_in_color_name(board)
    depth = 0
    applied = 0
    cached_actions = leaf_actions
    cached_priors = leaf_priors

    while True:
        terminal_value = _terminal_value_for(board, ply, perspective)
        if terminal_value is not None:
            return terminal_value, applied

        if depth >= depth_cap:
            return _eval_value_for(board, perspective), applied

        action = rollout_policy_action(board, cached_actions, cached_priors)
        if action is None:
            return 0.0, applied

        # Cache only valid for step 0 — board changes after apply.
        cached_actions = None
        cached_priors = None

        sim_actions.append((board.turn_color.name, action))
        board.apply(action)
        ply += 1
        depth += 1
        applied += 1


def backprop(leaf: Node, reward: float, sim_actions: list, applied_count: int) -> None:
    """Walk leaf→root, updating Q stats and RAVE on children of each ancestor.

    `sim_actions[:applied_count]` are the tree edges descended from root to leaf;
    `sim_actions[applied_count:]` are the rollout actions. At depth d, the actions
    "below" this node start at index d. Of those, every other entry was played
    by this node's side to move (colours strictly alternate), so we step by 2.
    """
    node = leaf
    depth = applied_count
    n_actions = len(sim_actions)

    while node is not None:
        node.visits += 1
        node.total_value += reward

        # RAVE: at this node, update children whose move was played later in the
        # simulation by this node's side. `reward` here is in "moved-into-node"
        # frame; the children's rave_value lives in the opposite frame
        # (parent's-side perspective), so flip.
        rave_reward = -reward
        for i in range(depth, n_actions, 2):
            action = sim_actions[i][1]
            child = node.children.get(action)
            if child is not None:
                child.rave_visits += 1
                child.rave_value += rave_reward

        reward = -reward
        depth -= 1
        node = node.parent


# def best_child(root: Node) -> Node | None:
#     """Robust child with a visit-cluster Q tiebreak (Tier 2.4).

#     Standard MCTS picks max-visits, ties broken by Q. The diagnostic showed
#     4/20 root decisions where the most-visited child had a *strictly worse*
#     Q than another sibling whose visit count was within a few percent —
#     PUCT explored a high-prior child more without that prior translating
#     to better Q. So: among children whose visit count is within 10 % of
#     the max, pick the one with the highest Q. This keeps the robustness
#     of the visit-count rule (avoid one-shot lucky rollouts) while letting
#     Q break statistically-tied visit counts.
#     """
#     if not root.children:
#         return None
#     max_v = max(c.visits for c in root.children.values())
#     threshold = max(1, max_v * 0.9)
#     cluster = [c for c in root.children.values() if c.visits >= threshold]
#     return max(
#         cluster,
#         key=lambda c: (c.total_value / c.visits) if c.visits > 0 else -math.inf,
#     )

def best_child(root: Node) -> Node | None:
    if not root.children:
        return None

    children = list(root.children.values())

    # If a proven win exists, always take it.
    wins = [c for c in children if c.terminal_value == 1.0]
    if wins:
        return max(wins, key=lambda c: c.visits)

    # Avoid proven losses if any non-losing option exists.
    non_losses = [c for c in children if c.terminal_value != -1.0]
    if non_losses:
        children = non_losses

    max_v = max(c.visits for c in children)
    threshold = max(1, max_v * 0.9)
    cluster = [c for c in children if c.visits >= threshold]

    return max(
        cluster,
        key=lambda c: (c.total_value / c.visits) if c.visits > 0 else -math.inf,
    )


def mcts_final(
    root_board,
    budget: TimeBudget,
    root: Node | None = None,
    c_puct: float = _C_PUCT,
    rollout_depth_cap: int = _ROLLOUT_DEPTH_CAP,
) -> tuple[Action | None, Node]:
    
    if root is None:
        root = Node(
            parent=None,
            incoming_move=None,
            board_hash=root_board.zobrist_hash,
    )
        
    root_win = _immediate_winning_move(root_board, 0)
    if root_win is not None:
        return root_win, root


    while not budget.expired():

        board = root_board

        node, ply, sim_actions = select(root, board, c_puct)
        node, ply, used_hybrid = expand(node, board, ply, sim_actions)
        applied_count = len(sim_actions)
        rollout_applied = 0

        if node.terminal_value is not None:
            reward = node.terminal_value
        elif used_hybrid:
            reward = node.total_value / node.visits
        else:
            # Pre-materialise the leaf's action+prior cache for reuse on the
            # rollout's first step. Cost is paid here regardless because the
            # leaf will need it on its next selection visit anyway.
            if node.untried is None:
                _materialise_actions(node, board)


            
            reward, rollout_applied = heavy_rollout(
                board,
                ply,
                sim_actions,
                rollout_depth_cap,
                leaf_actions=node.untried,
                leaf_priors=node.untried_rollout_weights,
            )

        backprop(node, reward, sim_actions, applied_count)
        cur = node
        while cur is not None:
            _try_prove(cur)
            cur = cur.parent

        for _ in range(rollout_applied):
            board.undo()

        for _ in range(applied_count):
            board.undo()



    # for move, child in root.children.items():
    #     q = (child.total_value / child.visits) if child.visits > 0 else 0.0
    #     print("ROOT", move, "visits=", child.visits, "q=", q, "term=", child.terminal_value)
    
    chosen = best_child(root)
    # print("CHOSEN", chosen.incoming_move if chosen is not None else None)



    return (chosen.incoming_move if chosen is not None else None), root

