from referee.game import PlayerColor, Action

from agent.core.board import GameState
from agent.core.placement import choose_placement_action
from agent.core.time_budget import TimeBudget, per_move_budget
from agent.search.mcts_final import mcts_final
from agent.search.pvs import iterative_deepening_pvs
from agent.core.tt import TranspositionTable


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self.state = GameState()
        self._root = None
        self._tt = TranspositionTable()

    def action(self, **referee: dict) -> Action:
        print("working111111111111111111111111111111111111")
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

        if time_remaining < 0.5:
            move = iterative_deepening_pvs(self.state, budget, self._tt, max_depth=2)
            print("used a-b fallback")
            if move is not None:
                self._root = None
                return move

        # move, self._root = mcts_final(self.state, budget, root=self._root)
        move, self._root = mcts_final(self.state, budget, root=None)

        if move is None:
            for action in self.state.legal_actions():
                move = action
                break

        if move is None:
            raise ValueError("No legal play action found")

        return move
    
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.state.apply(action)
        self._root = None

    # def update(self, color: PlayerColor, action: Action, **referee: dict):
    #     self.state.apply(action)

    #     if self._root is None:
    #         return

    #     child = self._root.children.get(action)
    #     if child is None:
    #         self._root = None
    #         return

    #     child.parent = None
    #     self._root = child

    #     assert self._root.board_hash == self.state.zobrist_hash