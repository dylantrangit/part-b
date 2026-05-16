# Uniform-random opponent: picks a random legal action each turn. Used as
# the lowest-rung baseline in performance evaluation (§5).

import random

from referee.game import PlayerColor, Action

from agent.core.board import GameState
from agent.core.placement import choose_placement_action


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self.state = GameState()
        self._rng = random.Random(0xC45CADE ^ (0 if color == PlayerColor.RED else 1))

    def action(self, **referee: dict) -> Action:
        if self.state.get_phase() == "placement":
            move = choose_placement_action(self.state)
            if move is not None:
                return move

        actions = list(self.state.legal_actions())
        if not actions:
            raise ValueError("No legal action found")
        return self._rng.choice(actions)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.state.apply(action)
