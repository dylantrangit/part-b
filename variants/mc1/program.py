from referee.game import PlayerColor, Action

from agent.core.board import GameState
from agent.core.placement import choose_placement_action
from agent.search.mcts_uct import mcts


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

        move = mcts(self.state, time_limit_s=0.95)

        if move is None:
            for action in self.state.legal_actions():
                move = action
                break

        if move is None:
            raise ValueError("No legal play action found")

        return move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.state.apply(action)