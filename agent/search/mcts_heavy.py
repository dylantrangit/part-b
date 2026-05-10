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

from referee.game import Action, PlayerColor

from ..core.eval import evaluate
from ..core.policy import (
    _SOFTMAX_TEMP as _TAU_ROLLOUT,
    _TAU_PRIOR,
    heuristic_score,
    rollout_policy_action,
)
from ..core.time_budget import TimeBudget


_RAVE_BIAS = 1e-5
_FPU_REDUCTION = 0.25
_EVAL_SCALE = 500.0
# Shorter rollouts so we (a) do more iterations per unit time and (b) lean
# more on the calibrated `evaluate()` cutoff than on the noisy rollout
# heuristic. 15 keeps enough lookahead for tactical sequences without
# compounding heuristic noise.
_ROLLOUT_DEPTH_CAP = 15
# PUCT exploration constant. The AlphaZero default (2.0) gives a sensible
# balance between Q-exploitation and prior-guided exploration at the 30–80
# branching factor of Cascade midgame.
_C_PUCT = 2.0
# Endgame boost: from this ply onward the rollout-cutoff eval starts
# leaning on raw token count more heavily, ramped to full strength at
# ENDGAME_FULL_PLY. Avoids drifting into turn-limit losses where I3's
# tactical depth wins on token diff.
_ENDGAME_BOOST_START_PLY = 200
_ENDGAME_BOOST_FULL_PLY = 280
_ENDGAME_BOOST_TOKEN_WEIGHT = 100.0
# Stalling penalty applied to root priors when we're ahead on tokens and
# a candidate move's apply state is already in play_history (i.e. heading
# toward 3-fold). Subtracted from the heuristic score before softmax.
_STALLING_PENALTY = 1.0


@dataclass
class Node:
    parent: "Node | None"
    incoming_move: Action | None
    # P(incoming_move) — softmax-normalised heuristic prior set at expansion
    # by the parent. Drives the PUCT exploration term in `_selection_score`.
    prior: float = 0.0
    children: dict[Action, "Node"] = field(default_factory=dict)
    untried: list[Action] | None = None
    # Softmax-normalised priors at τ_prior, aligned with `untried` and
    # sorted ascending so `pop()` returns the highest-prior action first.
    # Used as PUCT priors for selection.
    untried_priors: list[float] | None = None
    # Softmax weights at τ_rollout, also aligned with `untried`. Passed to
    # `heavy_rollout` for sampling the first rollout step. Stored separately
    # because τ_prior < τ_rollout — we want sharp priors for PUCT but
    # broad sampling for rollout diversity.
    untried_rollout_weights: list[float] | None = None
    visits: int = 0
    total_value: float = 0.0
    rave_visits: int = 0
    rave_value: float = 0.0
    terminal_value: float | None = None


def _moved_in_color_name(board) -> str:
    return "BLUE" if board.turn_color.name == "RED" else "RED"


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
    # Endgame boost: bias the rollout-cutoff signal toward token-diff as the
    # 300-turn limit approaches. The shared `evaluate()` already has a small
    # endgame_emphasis term (kicks in at ply 280); this is an MCTS-only
    # extra so it doesn't leak into PVS and benefit I3 too.
    if board.play_ply >= _ENDGAME_BOOST_START_PLY:
        boost_span = _ENDGAME_BOOST_FULL_PLY - _ENDGAME_BOOST_START_PLY
        ramp = min(1.0, (board.play_ply - _ENDGAME_BOOST_START_PLY) / boost_span)
        raw += ramp * _ENDGAME_BOOST_TOKEN_WEIGHT * (board.red_tokens - board.blue_tokens)
    if perspective == "BLUE":
        raw = -raw
    return math.tanh(raw / _EVAL_SCALE)


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

        chosen = max(
            node.children.values(),
            key=lambda child: _selection_score(child, node, c),
        )
        sim_actions.append((board.turn_color.name, chosen.incoming_move))
        board.apply(chosen.incoming_move)
        ply += 1
        node = chosen


def _materialise_with_actions(node: Node, board, actions, score_offsets=None) -> None:
    """Populate node action lists from an explicit `actions` list.

    Used both by the standard `_materialise_actions` (full legal list) and
    by `mcts()` to set up the root with a curated subset (filter
    decisive-loss moves, prune clearly-bad moves). `score_offsets`, if
    provided, are added to the heuristic scores before softmax — used at
    the root to demote stalling moves when we're ahead on tokens.
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
    # Sort ascending by raw score so .pop() (LIFO) returns the highest-
    # prior action first — front-loads expansion onto plausible moves.
    order = sorted(range(len(actions)), key=lambda i: scores[i])
    node.untried = [actions[i] for i in order]
    node.untried_priors = [priors[i] for i in order]
    node.untried_rollout_weights = [rollout_weights[i] for i in order]


def _materialise_actions(node: Node, board) -> None:
    """Populate untried + priors (τ_prior) + rollout weights (τ_rollout)
    for an internal-tree node, using the full legal-action list.
    """
    _materialise_with_actions(node, board, list(board.legal_actions()))


def _find_root_immediate_win(state, actions, is_red):
    """Return any action that drops opponent tokens to 0, or None."""
    enemy_attr = "blue_tokens" if is_red else "red_tokens"
    for action in actions:
        state.apply(action)
        won = getattr(state, enemy_attr) == 0
        state.undo()
        if won:
            return action
    return None


def _filter_root_safe_actions(state, actions, is_red):
    """Drop actions that let the opponent win on their next turn.

    Gated on us being in elimination range (own_stacks ≤ 2 or own_tokens ≤ 5)
    so the O(N²) apply/undo scan stays out of unthreatened mid-game positions.
    Returns the full list when not in range or when every move is losing
    (MCTS will then pick the least-bad option as before).
    """
    own_stacks = state.red_stacks if is_red else state.blue_stacks
    own_tokens = state.red_tokens if is_red else state.blue_tokens
    if own_stacks > 2 and own_tokens > 5:
        return actions

    own_attr = "red_tokens" if is_red else "blue_tokens"
    safe = []
    for action in actions:
        state.apply(action)
        opp_can_win = False
        for opp_action in state.legal_actions():
            state.apply(opp_action)
            opp_won = getattr(state, own_attr) == 0
            state.undo()
            if opp_won:
                opp_can_win = True
                break
        state.undo()
        if not opp_can_win:
            safe.append(action)
    return safe if safe else actions


def _stalling_penalties(state, actions, is_red):
    """Return one penalty per action that demotes 3-fold-stalling moves.

    Only fires when we're ahead on tokens — a draw by repetition would lose
    that material lead. A move whose apply state is already in play_history
    counts as a stalling move (one more visit gets us closer to 3-fold).
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
    """Drop clearly-bad actions (heuristic score < threshold).

    Reduces effective branching at the root so each remaining child gets
    proportionally more sims — opening Q estimates are noise-limited at
    ~15 sims/child.
    """
    if len(actions) <= min_keep + 2:
        return actions
    scored = [(a, heuristic_score(state, a)) for a in actions]
    kept = [a for a, s in scored if s >= threshold]
    if len(kept) < min_keep:
        return actions
    return kept


def expand(node: Node, board, ply: int, sim_actions: list):
    terminal_value = _terminal_value_for(board, ply, _moved_in_color_name(board))
    if terminal_value is not None:
        node.terminal_value = terminal_value
        return node, ply

    if node.untried is None:
        _materialise_actions(node, board)

    if not node.untried:
        return node, ply

    move = node.untried.pop()
    prior = node.untried_priors.pop() if node.untried_priors else 0.0
    if node.untried_rollout_weights is not None:
        node.untried_rollout_weights.pop()
    sim_actions.append((board.turn_color.name, move))
    board.apply(move)
    child = Node(parent=node, incoming_move=move, prior=prior)
    node.children[move] = child
    return child, ply + 1


def heavy_rollout(
    board,
    ply: int,
    sim_actions: list,
    depth_cap: int = _ROLLOUT_DEPTH_CAP,
    leaf_actions=None,
    leaf_priors=None,
) -> float:
    perspective = _moved_in_color_name(board)
    depth = 0
    cached_actions = leaf_actions
    cached_priors = leaf_priors

    while True:
        terminal_value = _terminal_value_for(board, ply, perspective)
        if terminal_value is not None:
            return terminal_value

        if depth >= depth_cap:
            return _eval_value_for(board, perspective)

        action = rollout_policy_action(board, cached_actions, cached_priors)
        if action is None:
            return 0.0

        # Cache only valid for step 0 — the board mutates after apply.
        cached_actions = None
        cached_priors = None

        sim_actions.append((board.turn_color.name, action))
        board.apply(action)
        ply += 1
        depth += 1


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


def best_child(root: Node) -> Node | None:
    """Robust child with a visit-cluster Q tiebreak.

    Standard MCTS picks max-visits, ties broken by Q. But two children with
    visits within a few percent are statistically tied, and PUCT can keep
    exploring a high-prior child without that prior translating to better
    Q. So: among children whose visit count is within 10 % of the max,
    pick the one with the highest Q. Preserves the robustness of the
    visit-count rule (avoid one-shot lucky rollouts) while letting Q
    break statistically-tied visit counts.
    """
    if not root.children:
        return None
    max_v = max(c.visits for c in root.children.values())
    threshold = max(1, max_v * 0.9)
    cluster = [c for c in root.children.values() if c.visits >= threshold]
    return max(
        cluster,
        key=lambda c: (c.total_value / c.visits) if c.visits > 0 else -math.inf,
    )


def mcts(
    root_board,
    budget: TimeBudget,
    c_puct: float = _C_PUCT,
    rollout_depth_cap: int = _ROLLOUT_DEPTH_CAP,
) -> Action | None:
    legal = list(root_board.legal_actions())
    if not legal:
        return None

    is_red = root_board.turn_color == PlayerColor.RED

    # Short-circuit on a forced win — eliminates the opponent in one move.
    # Avoids wasting MCTS budget when the answer is immediate.
    win_move = _find_root_immediate_win(root_board, legal, is_red)
    if win_move is not None:
        return win_move

    # Anti-decisive filter. Drop any move that leaves the opponent a winning
    # response next turn, when we're vulnerable enough for that to happen.
    actions_to_search = _filter_root_safe_actions(root_board, legal, is_red)

    # Prune clearly-bad actions to reduce effective branching at the root,
    # giving each remaining child more sims.
    actions_to_search = _pruned_actions(root_board, actions_to_search)

    # Stalling penalties: when ahead on tokens, demote moves whose post-
    # apply state is already in play_history (they push toward 3-fold draw,
    # which loses our material lead).
    stalling = _stalling_penalties(root_board, actions_to_search, is_red)

    root = Node(parent=None, incoming_move=None)
    _materialise_with_actions(root, root_board, actions_to_search, score_offsets=stalling)

    while not budget.expired():
        board = root_board

        node, ply, sim_actions = select(root, board, c_puct)
        node, ply = expand(node, board, ply, sim_actions)
        applied_count = len(sim_actions)

        if node.terminal_value is not None:
            reward = node.terminal_value
        else:
            # Pre-materialise the leaf's action+prior cache for reuse on the
            # rollout's first step. Cost is paid here regardless because the
            # leaf will need it on its next selection visit anyway.
            if node.untried is None:
                _materialise_actions(node, board)
            rollout_board = board.copy()
            reward = heavy_rollout(
                rollout_board,
                ply,
                sim_actions,
                rollout_depth_cap,
                leaf_actions=node.untried,
                # Rollout uses τ_rollout-temperature weights for sampling
                # diversity, not the τ_prior priors used by PUCT.
                leaf_priors=node.untried_rollout_weights,
            )

        backprop(node, reward, sim_actions, applied_count)

        for _ in range(applied_count):
            board.undo()

    chosen = best_child(root)
    return chosen.incoming_move if chosen is not None else None
