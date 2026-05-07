# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Action

from .core.board import GameState
from .core.placement import choose_placement_action
from .core.time_budget import TimeBudget, per_move_budget
from .core.tt import TranspositionTable
from .search.pvs import iterative_deepening_pvs as iterative_deepening


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self.state = GameState()
        self.tt = TranspositionTable()


    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """

        if self.state.turn_color != self._color:
            raise ValueError(
                f"wrong color : {self._color}"
            )

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
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        self.state.apply(action)
