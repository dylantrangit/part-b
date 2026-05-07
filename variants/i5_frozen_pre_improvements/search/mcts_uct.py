from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from referee.game import Action

from ..core.time_budget import TimeBudget


@dataclass
class Node:
    parent: "Node | None"
    incoming_move: Action | None
    children: dict[Action, "Node"] = field(default_factory=dict)
    untried: list[Action] | None = None
    visits: int = 0
    total_value: float = 0.0
    terminal_value: float | None = None


def _moved_in_color_name(board) -> str:
    """The colour of the player who just moved into the current state."""
    return "BLUE" if board.turn_color.name == "RED" else "RED"


def _terminal_value_for(board, ply: int, perspective: str) -> float | None:
    """Terminal reward from `perspective`'s POV in [-1, +1], or None if non-terminal.

    The MCTS convention used here is that a leaf's reward (and therefore each
    node's `total_value`) is from the perspective of the player who just moved
    INTO that leaf — equivalently, the parent's side to move. Selection then
    uses plain `max` and backprop flips sign at every step.
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


def uct_score(child: Node, parent_visits: int, c: float = math.sqrt(2)) -> float:
    if child.visits == 0:
        return math.inf

    return (
        child.total_value / child.visits
        + c * math.sqrt(math.log(parent_visits) / child.visits)
    )


def select(root: Node, board, c: float) -> tuple[Node, int, int]:
    node = root
    ply = 0
    applied_count = 0

    while True:
        terminal_value = _terminal_value_for(board, ply, _moved_in_color_name(board))
        if terminal_value is not None:
            node.terminal_value = terminal_value
            return node, ply, applied_count

        if node.untried is None:
            node.untried = list(board.legal_actions())

        if node.untried:
            return node, ply, applied_count

        if not node.children:
            return node, ply, applied_count

        node = max(
            node.children.values(),
            key=lambda child: uct_score(child, node.visits, c),
        )
        board.apply(node.incoming_move)
        ply += 1
        applied_count += 1


def expand(node: Node, board, ply: int, applied_count: int) -> tuple[Node, int, int]:
    terminal_value = _terminal_value_for(board, ply, _moved_in_color_name(board))
    if terminal_value is not None:
        node.terminal_value = terminal_value
        return node, ply, applied_count

    if node.untried is None:
        node.untried = list(board.legal_actions())

    if not node.untried:
        return node, ply, applied_count

    move = node.untried.pop()
    board.apply(move)
    child = Node(parent=node, incoming_move=move)
    node.children[move] = child
    return child, ply + 1, applied_count + 1


def rollout(board, ply: int, depth_cap: int = 300) -> float:
    # Lock in the leaf's "who-just-moved" colour up front. Returning the reward
    # in this fixed frame regardless of how many random moves the rollout plays
    # keeps the sign consistent with what backprop expects at the leaf.
    perspective = _moved_in_color_name(board)
    depth = 0

    while True:
        terminal_value = _terminal_value_for(board, ply, perspective)
        if terminal_value is not None:
            return terminal_value

        if depth >= depth_cap:
            return 0.0

        actions = list(board.legal_actions())
        if not actions:
            return 0.0

        move = random.choice(actions)
        board.apply(move)
        ply += 1
        depth += 1


def backprop(node: Node, reward: float) -> None:
    while node is not None:
        node.visits += 1
        node.total_value += reward
        reward = -reward
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
    rollout_depth_cap: int = 300,
) -> Action | None:
    root = Node(parent=None, incoming_move=None)

    while not budget.expired():
        board = root_board

        node, ply, applied_count = select(root, board, exploration)
        node, ply, applied_count = expand(node, board, ply, applied_count)

        if node.terminal_value is not None:
            reward = node.terminal_value
        else:
            rollout_board = board.copy()
            reward = rollout(rollout_board, ply, rollout_depth_cap)

        backprop(node, reward)

        for _ in range(applied_count):
            board.undo()

    chosen = best_child(root)
    return chosen.incoming_move if chosen is not None else None