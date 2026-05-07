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
# Shorter rollouts than the original 25 so we (a) do more iterations per
# unit time and (b) lean more on the calibrated `evaluate()` cutoff than on
# the noisy rollout heuristic. Plan §9.1 picked 25 conservatively; 15 keeps
# enough lookahead for tactical sequences without compounding heuristic noise.
_ROLLOUT_DEPTH_CAP = 15
# PUCT exploration constant. Plan §10.1(b) suggests 1.5–2.5; 2.0 is the
# AlphaZero default and gives a sensible balance between Q-exploitation and
# prior-guided exploration in the 30–80 branching factor of Cascade midgame.
_C_PUCT = 2.0


@dataclass
class Node:
    parent: "Node | None"
    incoming_move: Action | None
    # P(incoming_move) — softmax-normalised heuristic prior set at expansion
    # by the parent. Drives the PUCT exploration term in `_selection_score`.
    prior: float = 0.0
    children: dict[Action, "Node"] = field(default_factory=dict)
    untried: list[Action] | None = None
    # Softmax-normalised priors aligned with `untried`, sorted ascending so
    # `pop()` gives the highest-prior action first (PUCT-style expansion
    # ordering). Reused as cached weights for the rollout's first softmax.
    untried_priors: list[float] | None = None
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
    """Populate `node.untried` and `node.untried_priors` from `board`.

    Computes a heuristic score per action, applies softmax to get priors
    summing to 1, then sorts by raw score so `pop()` returns the highest-
    prior action first. The priors are reused both as PUCT weights for
    selection AND as direct sampling weights for the rollout's first step.
    """
    actions = list(board.legal_actions())
    if not actions:
        node.untried = []
        node.untried_priors = []
        return
    scores = [heuristic_score(board, a) for a in actions]
    max_s = max(scores)
    weights = [math.exp((s - max_s) / 2.0) for s in scores]  # τ matches policy
    total = sum(weights)
    if total <= 0.0:
        # Degenerate (shouldn't happen with finite scores) — uniform prior.
        priors = [1.0 / len(actions)] * len(actions)
    else:
        priors = [w / total for w in weights]
    # Sort ascending by raw score so .pop() (LIFO) returns the highest-
    # prior action — front-loads expansion onto plausible moves.
    order = sorted(range(len(actions)), key=lambda i: scores[i])
    node.untried = [actions[i] for i in order]
    node.untried_priors = [priors[i] for i in order]


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
    c_puct: float = _C_PUCT,
    rollout_depth_cap: int = _ROLLOUT_DEPTH_CAP,
) -> Action | None:
    root = Node(parent=None, incoming_move=None)

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
                leaf_priors=node.untried_priors,
            )

        backprop(node, reward, sim_actions, applied_count)

        for _ in range(applied_count):
            board.undo()

    chosen = best_child(root)
    return chosen.incoming_move if chosen is not None else None
