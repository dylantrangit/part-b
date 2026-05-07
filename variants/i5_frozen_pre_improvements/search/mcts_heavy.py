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
from ..core.policy import heuristic_score, rollout_policy_action
from ..core.time_budget import TimeBudget


_RAVE_BIAS = 1e-5
_FPU_REDUCTION = 0.25
_EVAL_SCALE = 500.0
_ROLLOUT_DEPTH_CAP = 25


@dataclass
class Node:
    parent: "Node | None"
    incoming_move: Action | None
    children: dict[Action, "Node"] = field(default_factory=dict)
    untried: list[Action] | None = None
    # Heuristic prior per legal action, paired with `untried` (and the popped
    # children — popping `untried` also pops `untried_scores` to keep them
    # aligned). Computed once per node when its action list is materialised,
    # then reused at the leaf to skip recomputation on the first rollout step.
    # Also primes I6's PUCT prior.
    untried_scores: list[float] | None = None
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
    if perspective == "BLUE":
        raw = -raw
    return math.tanh(raw / _EVAL_SCALE)


def _selection_score(child: Node, parent: Node, c: float) -> float:
    if child.visits == 0:
        # FPU: estimate unvisited children from the parent's running Q.
        # parent.total_value is in "moved-into-parent" frame, i.e. the OPPOSITE
        # of the parent's side to move. Negate so we compare in the same frame
        # the visited siblings live in (parent-player perspective).
        if parent.visits > 0:
            parent_q = -parent.total_value / parent.visits
        else:
            parent_q = 0.0
        return parent_q - _FPU_REDUCTION

    n = child.visits
    q = child.total_value / n

    if child.rave_visits > 0:
        m = child.rave_visits
        rave_q = child.rave_value / m
        beta = m / (m + n + 4.0 * m * n * _RAVE_BIAS)
        q = (1.0 - beta) * q + beta * rave_q

    explore = c * math.sqrt(math.log(max(parent.visits, 1)) / n)
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


def _materialise_actions(node: Node, board) -> None:
    """Populate `node.untried` and `node.untried_scores` from `board`.

    Kept paired so callers that pop `untried` also pop `untried_scores` and
    the indices stay aligned. The full pre-pop list is what the rollout
    consumes as its cached prior at the leaf, so this materialisation pays
    for itself the first time the node is rolled out from.
    """
    actions = list(board.legal_actions())
    node.untried = actions
    node.untried_scores = [heuristic_score(board, a) for a in actions] if actions else []


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
    if node.untried_scores is not None:
        node.untried_scores.pop()
    sim_actions.append((board.turn_color.name, move))
    board.apply(move)
    child = Node(parent=node, incoming_move=move)
    node.children[move] = child
    return child, ply + 1


def heavy_rollout(
    board,
    ply: int,
    sim_actions: list,
    depth_cap: int = _ROLLOUT_DEPTH_CAP,
    leaf_actions=None,
    leaf_scores=None,
) -> float:
    perspective = _moved_in_color_name(board)
    depth = 0
    cached_actions = leaf_actions
    cached_scores = leaf_scores

    while True:
        terminal_value = _terminal_value_for(board, ply, perspective)
        if terminal_value is not None:
            return terminal_value

        if depth >= depth_cap:
            return _eval_value_for(board, perspective)

        action = rollout_policy_action(board, cached_actions, cached_scores)
        if action is None:
            return 0.0

        # Cache only valid for step 0 — the board mutates after apply.
        cached_actions = None
        cached_scores = None

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
    if not root.children:
        return None
    return max(
        root.children.values(),
        key=lambda child: (
            child.visits,
            (child.total_value / child.visits) if child.visits > 0 else -math.inf,
        ),
    )


def mcts(
    root_board,
    budget: TimeBudget,
    exploration: float = math.sqrt(2),
    rollout_depth_cap: int = _ROLLOUT_DEPTH_CAP,
) -> Action | None:
    root = Node(parent=None, incoming_move=None)

    while not budget.expired():
        board = root_board

        node, ply, sim_actions = select(root, board, exploration)
        node, ply = expand(node, board, ply, sim_actions)
        applied_count = len(sim_actions)

        if node.terminal_value is not None:
            reward = node.terminal_value
        else:
            # Pre-materialise the leaf's action+score cache for reuse on the
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
                leaf_scores=node.untried_scores,
            )

        backprop(node, reward, sim_actions, applied_count)

        for _ in range(applied_count):
            board.undo()

    chosen = best_child(root)
    return chosen.incoming_move if chosen is not None else None
