# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from math import inf

from referee.game import PlayerColor, Action

from .board import GameState
from .placement import choose_placement_action
from .search import negamax_fixed


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

        _, move = negamax_fixed(self.state, depth=3, alpha=-inf, beta=inf)
        if move is None:
            raise ValueError("No legal play action found")

        return move
    

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        self.state.apply(action)

        # There are four possible action types: PLACE, MOVE, EAT, and CASCADE.
        # Below we check which type of action was played and print out the
        # details of the action for demonstration purposes. You should replace
        # this with your own logic to update your agent's internal game state.

