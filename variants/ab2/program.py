# Variant: iterative-deepening alpha-beta + TT + move ordering (Iteration 2).
# Mirrors the current shipping agent (agent.program) but lives here so the
# benchmarking harness can run it as a distinct module.

from referee.game import PlayerColor, Action

from agent.core.board import GameState
from agent.core.placement import choose_placement_action
from agent.core.time_budget import TimeBudget, per_move_budget
from agent.core.tt import TranspositionTable
from agent.search.ab_id import iterative_deepening


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self.state = GameState()
        self.tt = TranspositionTable()

    def action(self, **referee: dict) -> Action:
        if self.state.turn_color != self._color:
            raise ValueError(f"wrong color : {self._color}")

        if self.state.get_phase() == "placement":
            move = choose_placement_action(self.state)
            if move is None:
                raise ValueError("No legal placement action found")
            return move

        time_remaining = referee.get("time_remaining")
        if time_remaining is None:
            time_remaining = 60.0

        plies_left = max(30, 300 - self.state.play_ply)
        moves_left = max(1, plies_left // 2)
        budget_seconds = per_move_budget(time_remaining, moves_left)
        budget = TimeBudget(budget_seconds)

        move = iterative_deepening(self.state, budget, self.tt)
        if move is None:
            for action in self.state.legal_actions():
                move = action
                break
        if move is None:
            raise ValueError("No legal play action found")
        return move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.state.apply(action)
