"""
Principal-variation stability across iterative-deepening depths.

For a given position, run α–β at depths 1..D using agent.search.pvs._pvs
directly (no agent-code modification). At each depth record the best move
and best value. Then report:

    - per-depth best-move sequence
    - PV-stability: fraction of depths whose best move == final-depth best
      (a noisy search flickers between candidates; a stable one converges)
    - depth at which the final best move is first seen
    - value drift between consecutive depths

A stable PV means the eval is correctly ranking the top candidate; a wild
PV means either the eval is noisy or the search is going too shallow.

Example:
    python tools/pv_stability.py --max-depth 7 -t 30 --positions 5
"""

import argparse
import os
import random
import sys
import time
from math import inf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.board import GameState
from agent.core.placement import choose_placement_action
from agent.core.time_budget import TimeBudget
from agent.core.tt import TranspositionTable
from agent.search.pvs import _pvs


def _random_midgame(seed: int, play_plies: int) -> GameState:
    """Build a midgame position: play 8 placements + `play_plies` random play moves."""
    rng = random.Random(seed)
    state = GameState()
    # Placement phase via the agent's own placement policy (gives realistic openings).
    for _ in range(8):
        move = choose_placement_action(state)
        if move is None:
            break
        state.apply(move)
    # Random play moves to reach a midgame.
    for _ in range(play_plies):
        if state.terminal() is not None:
            break
        legal = list(state.legal_actions())
        if not legal:
            break
        state.apply(rng.choice(legal))
    return state


def per_depth_search(state: GameState, max_depth: int, time_limit: float):
    """Run α–β at depth 1..max_depth, capturing the best move and value at each."""
    tt = TranspositionTable()
    killers: dict = {}
    history: dict = {}
    budget = TimeBudget(time_limit)
    pv_records: list[tuple[int, object, float, float]] = []   # depth, move, val, elapsed
    start = time.monotonic()
    for d in range(1, max_depth + 1):
        d_start = time.monotonic()
        try:
            val, move = _pvs(state, d, -inf, inf, 0, budget, tt, killers, history)
        except Exception as e:
            print(f"  depth {d}: timeout/exception ({type(e).__name__})")
            break
        pv_records.append((d, move, val, time.monotonic() - d_start))
        if budget.expired():
            break
    return pv_records, time.monotonic() - start


def format_move(move):
    if move is None:
        return "(none)"
    cls = type(move).__name__.replace("Action", "")
    return f"{cls}({move.coord.r},{move.coord.c}" + (f",{move.direction.name})" if hasattr(move, "direction") else ")")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("-t", "--time-limit", type=float, default=30.0,
                        help="per-position time limit so the search can reach max-depth")
    parser.add_argument("--positions", type=int, default=3,
                        help="how many random midgame positions to test")
    parser.add_argument("--play-plies", type=int, default=20,
                        help="random play plies after placement to reach midgame")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    overall_start = time.monotonic()
    stabilities = []

    for pi in range(args.positions):
        seed = rng.randrange(10 ** 9)
        state = _random_midgame(seed, args.play_plies)
        print(f"\n=== Position {pi+1}/{args.positions} (seed={seed}, ply={state.play_ply}) ===")
        records, elapsed = per_depth_search(state, args.max_depth, args.time_limit)
        if not records:
            print("  no depths completed")
            continue
        final_d, final_move, final_val, _ = records[-1]
        match_final = sum(1 for d, m, _v, _e in records if m == final_move)
        stability = match_final / len(records)
        stabilities.append(stability)
        # First depth that locks onto the final answer:
        lock_in = None
        for d, m, _v, _e in records:
            if m == final_move:
                lock_in = d
                break
        prev_val = None
        print(f"  {'depth':>5} {'move':<32} {'value':>9} {'Δval':>8} {'time(s)':>7}")
        for d, m, v, e in records:
            dv = "—" if prev_val is None else f"{v - prev_val:+.0f}"
            mark = " *" if m == final_move else ""
            print(f"  {d:5d} {format_move(m):<32} {v:9.0f} {dv:>8} {e:7.2f}{mark}")
            prev_val = v
        print(f"  stability: {match_final}/{len(records)} = {stability*100:.0f}%   "
              f"lock-in depth: {lock_in}")
        print(f"  total search time: {elapsed:.1f}s, max depth reached: {final_d}")

    if stabilities:
        avg = sum(stabilities) / len(stabilities)
        print()
        print(f"Average PV stability across {len(stabilities)} positions: {avg*100:.0f}%")
    print(f"Wall: {(time.monotonic()-overall_start)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
