"""I6 MCTS backend.

Builds on the I5 heavy rollout skeleton (`mcts_heavy.py`) and adds:
  - terminal proof propagation via def _terminal_value_for(board, ply: int, perspective: str) -> float | None:
  - also added colour asymmetric constants for PUCT and rollout depth caps

Reward convention (same as I5): a leaf's reward is in the perspective of the
player who just moved INTO the leaf. Each node's `total_value` therefore stores
from "moved-into-self" perspective; `backprop` add-then-flips. RAVE rewards on
a child are stored in the same frame so they combine with Q without sign games.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field

from referee.game import Action, PlayerColor, EatAction

from ..core.eval import evaluate
from ..core.policy import (
    _SOFTMAX_TEMP as _TAU_ROLLOUT,
    _TAU_PRIOR,
    heuristic_score,
    rollout_policy_action,
)
from ..core.time_budget import TimeBudget


from ..core.tt import TranspositionTable



_RAVE_BIAS = 1e-5
_FPU_REDUCTION = 0.25
_EVAL_SCALE = 500.0

#depth cap for blue, we change to 12 on red
_ROLLOUT_DEPTH_CAP = 9
# PUCT exploration constant. Plan §10.1(b) suggests 1.5–2.5; 2.0 is the
# AlphaZero default and gives a sensible balance between Q-exploitation and
# prior-guided exploration in the 30–80 branching factor of Cascade midgame.
_C_PUCT = 1.5

#comment ts later


_ENDGAME_BOOST_START_PLY = 200
_ENDGAME_BOOST_FULL_PLY = 280
_ENDGAME_BOOST_TOKEN_WEIGHT = 100.0
_STALLING_PENALTY = 1.0



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
    """MCTS-Solver proof propagation.
    Once all children of a node are resolved, the node itself can sometimes be
    proven. If any child is a proven loss for the opponent, this node is a proven
    win. If every child is a proven win for the opponent, this node is a proven
    loss.
    """
    if node.terminal_value is not None:
        return
    if node.untried is None or node.untried:
        return
    if not node.children:
        return

    vals = [child.terminal_value for child in node.children.values()]
    if any(v is None for v in vals):
        return

    if any(v == -1.0 for v in vals):
        node.terminal_value = 1.0
    elif all(v == 1.0 for v in vals):
        node.terminal_value = -1.0
    else:
        node.terminal_value = 0.0


def _terminal_value_for(board, ply: int, perspective: str) -> float | None:
    """Return terminal value based on colour perspective
    """
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
    """evaluate a non-terminal leaf from colour perspective.
    The raw heuristic eval is RED-positive, so it is negated for BLUE. tanh keeps
    the returned value bounded in the same range as terminal outcomes.
    """
    raw = evaluate(board)
    if board.play_ply >= _ENDGAME_BOOST_START_PLY:
        boost_span = _ENDGAME_BOOST_FULL_PLY - _ENDGAME_BOOST_START_PLY
        ramp = min(1.0, (board.play_ply - _ENDGAME_BOOST_START_PLY) / boost_span)
        raw += ramp * _ENDGAME_BOOST_TOKEN_WEIGHT * (board.red_tokens - board.blue_tokens)
    if perspective == "BLUE":
        raw = -raw
    return math.tanh(raw / _EVAL_SCALE)


def _find_root_immediate_win(state, actions, is_red):
    """checks if there is any root action which immediatley wins the match
    """
    enemy_attr = "blue_tokens" if is_red else "red_tokens"
    for action in actions:
        state.apply(action)
        won = getattr(state, enemy_attr) == 0
        state.undo()
        if won:
            return action
    return None



def _stalling_penalties(state, actions, is_red):
    """Penalises repetitive movements when ahead in material
    """
    own_tokens = state.red_tokens if is_red else state.blue_tokens
    enemy_tokens = state.blue_tokens if is_red else state.red_tokens
    if own_tokens <= enemy_tokens:
        return [0.0] * len(actions)

    penalties = []
    for action in actions:
        state.apply(action)
        already_seen = state.play_history.get(int(state.zobrist_hash), 0) >= 1
        state.undo()
        penalties.append(-_STALLING_PENALTY if already_seen else 0.0)
    return penalties


def _pruned_actions(state, actions, threshold=-0.5, min_keep=2):
    """Removes obviously poor actions according to our heuristic evaluaation

    also implemented a fallback feature to the full action list to avoid
    over-pruning when there are too few actions.
    """
    if len(actions) <= min_keep + 2:
        return actions
    scored = [(a, heuristic_score(state, a)) for a in actions]
    kept = [a for a, s in scored if s >= threshold]
    if len(kept) < min_keep:
        return actions
    return kept


def _selection_score(child: Node, parent: Node, c_puct: float) -> float:
    """PUCT score: Q(child) + c_puct · P(child) · √N_parent / (1 + n_child).

    The heuristic prior breaks UCB1's "every child gets explored uniformly
    first" pattern, which otherwise burns budget on plausibly-terrible
    openings before exploring promising lines.
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
    """Select the next child using proven outcomes first, then PUCT.

    in other words, use terminal vals over PUCT where possible
    """
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


def _materialise_with_actions(node: Node, board, actions, score_offsets=None) -> None:
    """Generate child action lists and convert heuristic scores into priors.
    same as I5
    """
    if not actions:
        node.untried = []
        node.untried_priors = []
        node.untried_rollout_weights = []
        return

    scores = [heuristic_score(board, a) for a in actions]
    if score_offsets is not None:
        scores = [s + o for s, o in zip(scores, score_offsets)]

    max_s = max(scores)
    prior_weights = [math.exp((s - max_s) / _TAU_PRIOR) for s in scores]
    total = sum(prior_weights)
    if total <= 0.0:
        priors = [1.0 / len(actions)] * len(actions)
    else:
        priors = [w / total for w in prior_weights]
    rollout_weights = [math.exp((s - max_s) / _TAU_ROLLOUT) for s in scores]

    order = sorted(range(len(actions)), key=lambda i: scores[i])
    node.untried = [actions[i] for i in order]
    node.untried_priors = [priors[i] for i in order]
    node.untried_rollout_weights = [rollout_weights[i] for i in order]


def _materialise_actions(node: Node, board) -> None:
    _materialise_with_actions(node, board, list(board.legal_actions()))



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

    node.children[move] = child
    return child, ply + 1, used_hybrid



def heavy_rollout(
    board,
    ply: int,
    sim_actions: list,
    depth_cap: int = _ROLLOUT_DEPTH_CAP,
    leaf_actions=None,
    leaf_priors=None,
) -> tuple[float, int]:
    
    """Run a heuristic-guided rollout from the current leaf.

    the rollout policy biases toward tactically plausible actions. If the depth cap is reached, the rollout
    stops and uses the heuristic evaluation as a leaf estimate.
    """
    
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
    """Walk leaf to root, updating Q stats and RAVE and term values on children of each ancestor.

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



def best_child(root: Node) -> Node | None:
    """Choose the final move from the root.

    Proven wins are preferred, proven losses are avoided, and otherwise the agent
    chooses the highest-Q child among the most-visited visit cluster.
    """
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
    # hybrid_value: float | None = None
) -> tuple[Action | None, Node]:
    
    """Run PUCT-guided heavy-rollout with terminal proofs MCTS until the time budget expires.

    The root is rebuilt each move because tree reuse was disabled after testing.
    The search first checks for immediate wins, prunes weak root actions, applies
    repetition penalties, then repeatedly performs selection, expansion, rollout,
    backpropagation, and terminal proof propagation.
    """

    iters = 0

    # If caller passed an old tree that does not match this board,
    # throw it away. Otherwise MCTS searches from the wrong position.
    # if root is not None and root.board_hash != root_board.zobrist_hash:
    #     root = None

    # if root is None:
    root = Node(
        parent=None,
        incoming_move=None,
        board_hash=root_board.zobrist_hash,
)
        
    legal = list(root_board.legal_actions())
    if not legal:
        return None, root

    is_red = root_board.turn_color == PlayerColor.RED

    if rollout_depth_cap == _ROLLOUT_DEPTH_CAP:
        rollout_depth_cap = 15 if is_red else 9


    win_move = _find_root_immediate_win(root_board, legal, is_red)
    if win_move is not None:
        return win_move, root
    

    actions_to_search = _pruned_actions(root_board, legal)


    stalling = _stalling_penalties(root_board, actions_to_search, is_red)

    if root.untried is None and not root.children:
        _materialise_with_actions(
            root,
            root_board,
            actions_to_search,
            score_offsets=stalling,
        )
            

    while not budget.expired():
        iters += 1
        board = root_board

        node, ply, sim_actions = select(root, board, c_puct)
        node, ply, used_hybrid = expand(node, board, ply, sim_actions)
        applied_count = len(sim_actions)
        rollout_applied = 0

        if node.terminal_value is not None:
            reward = node.terminal_value

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

    
    chosen = best_child(root)

    return (chosen.incoming_move if chosen is not None else None), root

