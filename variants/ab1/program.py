# Variant: depth-3 fixed-depth alpha-beta (Iteration 1 from the plan).
# Wires the shared agent.core state to the agent.search.ab_fixed backend so
# it benchmarks like-for-like against other variants.

from math import inf

from referee.game import PlayerColor, Action

from agent.core.board import GameState
from agent.core.placement import choose_placement_action
from agent.search.ab_fixed import negamax_fixed


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self.state = GameState()

    def action(self, **referee: dict) -> Action:
        if self.state.turn_color != self._color:
            raise ValueError(f"wrong color : {self._color}")

        if self.state.get_phase() == "placement":
            move = choose_placement_action(self.state)
            if move is None:
                raise ValueError("No legal placement action found")
            return move

        _, move = negamax_fixed(self.state, depth=3, alpha=-inf, beta=inf)
        if move is None:
            for action in self.state.legal_actions():
                move = action
                break
        if move is None:
            raise ValueError("No legal play action found")
        return move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.state.apply(action)
