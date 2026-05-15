"""
Tactical puzzle runner.

For each position in tests/tactical/positions.py, build a GameState
directly (skipping the placement phase), run a chosen search backend at a
fixed time budget, and check whether the move it picks is in the
position's accepted-best set (or, for blunder-avoidance puzzles, not in
the `forbid` set).

Reports per-puzzle pass/fail, the move actually chosen, and an overall
score. Catches regressions that a noisy 100-game round-robin can hide:
"agent X wins 53/100 on average but blunders the same forced eat we built
the eval to handle".

Example:
    python tools/tactical_runner.py --backend mcts -t 2
    python tools/tactical_runner.py --backend pvs  -t 5 --positions eat-equal-height mate-in-1
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from referee.game import PlayerColor

from agent.core.board import GameState
from agent.core.time_budget import TimeBudget
from agent.core.tt import TranspositionTable

# Search backends — imported lazily so a broken backend doesn't block the rest.
def _import_backend(name: str):
    if name == "pvs":
        from agent.search.pvs import iterative_deepening_pvs
        def go(state, t, tt):
            return iterative_deepening_pvs(state, TimeBudget(t), tt)
        return go
    if name == "mcts" or name == "mcts_heavy":
        from agent.search.mcts_heavy import mcts
        def go(state, t, _tt):
            return mcts(state, TimeBudget(t))
        return go
    if name == "mcts_final":
        from agent.search.mcts_final import mcts
        def go(state, t, _tt):
            return mcts(state, TimeBudget(t))
        return go
    if name == "ab":
        from agent.search.ab_id import iterative_deepening_ab
        def go(state, t, tt):
            return iterative_deepening_ab(state, TimeBudget(t), tt)
        return go
    raise ValueError(f"unknown backend: {name}")


def build_state(puzzle) -> GameState:
    state = GameState()
    # Plant pieces directly.
    from referee.game import Coord
    for (r, c), height in puzzle["pieces"].items():
        state.set_cell(Coord(r, c), height)
    # Bypass placement: skip to play phase. Per spec, placement_count >= 8
    # makes get_phase() == "play".
    state.placement_count = 8
    state.turn_color = puzzle["turn"]
    state.play_ply = puzzle.get("play_ply", 0)
    # Side-to-move zobrist key must match the turn color. The hash was
    # initialised assuming RED-to-move; flip if it's BLUE's turn.
    if puzzle["turn"] == PlayerColor.BLUE:
        from agent.core.zobrist import SIDE_TO_MOVE_KEY
        state.zobrist_hash ^= SIDE_TO_MOVE_KEY
    return state


def format_move(move):
    if move is None:
        return "(none)"
    cls = type(move).__name__.replace("Action", "")
    base = f"{cls}({move.coord.r},{move.coord.c}"
    if hasattr(move, "direction"):
        base += f",{move.direction.name}"
    return base + ")"


def actions_equal(a, b):
    if type(a) is not type(b):
        return False
    if getattr(a, "coord", None) != getattr(b, "coord", None):
        return False
    if getattr(a, "direction", None) != getattr(b, "direction", None):
        return False
    return True


def evaluate_puzzle(puzzle, move):
    """Return (passed: bool, why: str)."""
    forbid = puzzle.get("forbid", [])
    if any(actions_equal(move, f) for f in forbid):
        return False, f"played a forbidden move: {format_move(move)}"
    best = puzzle.get("best", [])
    if not best:
        # Pure blunder-avoidance puzzle: passing == not playing a forbidden move.
        return True, "(blunder avoided)"
    if any(actions_equal(move, b) for b in best):
        return True, "(matched expected best)"
    expected = ", ".join(format_move(b) for b in best)
    return False, f"expected one of {{{expected}}}"


def main():
    from tests.tactical.positions import PUZZLES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="pvs", choices=["pvs", "mcts", "mcts_heavy", "mcts_final", "ab"])
    parser.add_argument("-t", "--time-limit", type=float, default=2.0)
    parser.add_argument("--positions", nargs="*", help="run only these puzzle names")
    parser.add_argument("--tags", nargs="*", help="run only puzzles with one of these tags")
    args = parser.parse_args()

    search = _import_backend(args.backend)
    selected = PUZZLES
    if args.positions:
        selected = [p for p in selected if p["name"] in args.positions]
    if args.tags:
        selected = [p for p in selected if p.get("tag") in args.tags]
    if not selected:
        print("No puzzles selected.")
        return 1

    print(f"Backend: {args.backend}   t={args.time_limit}s   puzzles={len(selected)}")
    print("-" * 70)
    pass_count = 0
    overall_start = time.monotonic()
    for puzzle in selected:
        state = build_state(puzzle)
        tt = TranspositionTable()
        start = time.monotonic()
        try:
            move = search(state, args.time_limit, tt)
        except Exception as e:
            print(f"  [{puzzle['name']:<30}] EXCEPTION {type(e).__name__}: {e}")
            continue
        elapsed = time.monotonic() - start
        passed, why = evaluate_puzzle(puzzle, move)
        marker = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        print(f"  [{puzzle['name']:<30}] {marker}  played={format_move(move):<30} ({elapsed:.2f}s)  {why}")
        if not passed:
            print(f"     description: {puzzle['desc']}")

    elapsed = time.monotonic() - overall_start
    print("-" * 70)
    pct = 100 * pass_count / len(selected)
    print(f"Score: {pass_count}/{len(selected)} = {pct:.0f}%   ({elapsed:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
