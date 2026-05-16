# Greedy opponent: 1-ply lookahead over legal actions, scored by token-diff
# from the moving side's perspective. No adversarial recursion. Used as the
# second-rung baseline in performance evaluation (§5).

import random

from referee.game import PlayerColor, Action

from agent.core.board import GameState
from agent.core.placement import choose_placement_action


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self.state = GameState()
        self._rng = random.Random(0x6322D ^ (0 if color == PlayerColor.RED else 1))

    def action(self, **referee: dict) -> Action:
        if self.state.get_phase() == "placement":
            move = choose_placement_action(self.state)
            if move is not None:
                return move

        sign = 1 if self._color == PlayerColor.RED else -1
        best_score = None
        best_moves = []
        for action in self.state.legal_actions():
            self.state.apply(action)
            score = sign * (self.state.red_tokens - self.state.blue_tokens)
            self.state.undo()
            if best_score is None or score > best_score:
                best_score = score
                best_moves = [action]
            elif score == best_score:
                best_moves.append(action)

        if not best_moves:
            raise ValueError("No legal action found")
        return self._rng.choice(best_moves)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.state.apply(action)
