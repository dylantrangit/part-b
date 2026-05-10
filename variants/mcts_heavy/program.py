from referee.game import PlayerColor, Action

from agent.core.board import GameState
from agent.core.placement import choose_placement_action
from agent.core.time_budget import TimeBudget, per_move_budget
from agent.core.tt import TranspositionTable
from agent.search.mcts_heavy import mcts
from agent.search.pvs import iterative_deepening_pvs


# PVS handover thresholds. Diagnostic showed MCTS is noise-limited at ~15
# sims/child in the opening (very wide root branching) and tactically out-
# searched by PVS in the endgame. Use PVS for those regimes; MCTS owns the
# midgame where its long-horizon rollouts pay off.
#
# Opening dispatch is RED-only: a 12-game diagnostic showed adding PVS at
# ply<8 was +2 Red wins but −1 Blue win vs Tier 1 (MCTS opening). Hypothesis:
# PVS-vs-PVS opening sharpens Red's first-mover edge into a position MCTS
# struggles to defend, whereas Red's MCTS-as-attacker handles the same
# transition fine. Endgame dispatch stays for both (both colours benefit
# from PVS's tactical conversion under the 300-turn timer).
_OPENING_PVS_PLY = 8       # play_ply < 8 → first 4 play moves per side
_ENDGAME_PVS_PLY = 280     # play_ply >= 280 → last 20 plies before turn cap


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

        ply = self.state.play_ply
        opening_pvs = self._color == PlayerColor.RED and ply < _OPENING_PVS_PLY
        endgame_pvs = ply >= _ENDGAME_PVS_PLY
        if opening_pvs or endgame_pvs:
            move = iterative_deepening_pvs(self.state, budget, self.tt)
        else:
            move = mcts(self.state, budget)

        if move is None:
            for action in self.state.legal_actions():
                move = action
                break

        if move is None:
            raise ValueError("No legal play action found")

        return move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.state.apply(action)