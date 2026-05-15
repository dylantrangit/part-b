"""
Round-robin benchmark runner: pit any two referee-compatible modules
against each other for N games with balanced colours, run J in parallel.

Examples:
    python tools/bench.py --red agent --blue variants.ab1 -n 100 -j 4
    python tools/bench.py --red variants.ab2 --blue variants.ab1 -n 50 -t 30
    python tools/bench.py --red agent --blue variants.mcts_final --paired -n 50 -j 4

--paired mode pairs games so each ordinal (A as Red / B as Red) is run as
a back-to-back two-game mini-match. Reduces variance from first-mover
advantage and lets us report a "paired win" rate (A wins both or A wins
one + draws one) alongside the raw win rate.
"""

import argparse
import sys
import time

from _runner import run_pairings


def build_pairings(a: str, b: str, n: int, paired: bool):
    """Return list of (red, blue) game pairings."""
    if paired:
        half = n // 2
        out = []
        for _ in range(half):
            out.append((a, b))
            out.append((b, a))
        return out
    half = n // 2
    out = [(a, b)] * half
    out += [(b, a)] * (n - half)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--red", required=True, help="module path for agent A (e.g. agent, variants.ab1)")
    parser.add_argument("--blue", required=True, help="module path for agent B")
    parser.add_argument("-n", "--games", type=int, default=100, help="total games (split per side)")
    parser.add_argument("-j", "--parallel", type=int, default=4, help="concurrent games")
    parser.add_argument("-t", "--time-limit", type=float, default=60.0, help="seconds per agent per game")
    parser.add_argument("--paired", action="store_true",
                        help="run paired games (each ordinal played both ways back-to-back)")
    args = parser.parse_args()

    a, b = args.red, args.blue
    if a == b:
        print("warning: red and blue are the same module")
    pairings = build_pairings(a, b, args.games, args.paired)

    a_wins = 0
    b_wins = 0
    draws = 0
    failures = 0

    print(f"Running {len(pairings)} games ({'paired' if args.paired else 'unpaired'}) "
          f"at {args.parallel}-way parallelism, -t {args.time_limit}s/agent.")
    print(f"Agent A:  {a}")
    print(f"Agent B:  {b}")
    overall_start = time.monotonic()

    completed = 0
    for res in run_pairings(pairings, args.parallel, args.time_limit):
        completed += 1
        if res.draw:
            draws += 1
            outcome = "draw"
        elif res.winner == a:
            a_wins += 1
            outcome = f"{a} win"
        elif res.winner == b:
            b_wins += 1
            outcome = f"{b} win"
        else:
            failures += 1
            outcome = f"FAILED rc={res.rc}"
        print(
            f"[{completed:3d}/{len(pairings)}] "
            f"{res.red:>20} vs {res.blue:<20} -> {outcome:<24} "
            f"({res.elapsed:.1f}s)  "
            f"A={a_wins} B={b_wins} D={draws} F={failures}",
            flush=True,
        )

    overall_elapsed = time.monotonic() - overall_start
    total = a_wins + b_wins + draws + failures
    decisive = a_wins + b_wins
    print()
    print("=" * 72)
    print(f"Final: A={a} {a_wins} | B={b} {b_wins} | draws {draws} | failures {failures}")
    if total:
        print(f"  A win rate: {a_wins}/{total} = {100 * a_wins / total:.1f}%")
        print(f"  A score (W+0.5D): {(a_wins + 0.5 * draws) / total:.3f}")
        if decisive:
            print(f"  A win rate (decisive only): {a_wins}/{decisive} = {100 * a_wins / decisive:.1f}%")

    if args.paired:
        # Group by paired index; require both red and blue passes to compute paired stats.
        # Pairings are stored in order, but results may come back out of order — bench
        # printed them as completed. Paired-summary is approximate here; for strict
        # paired analysis use tools/sprt.py which guards on the pair.
        print(f"  (paired layout used: {len(pairings)//2} two-game pairs)")
    print(f"Wall time: {overall_elapsed/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
