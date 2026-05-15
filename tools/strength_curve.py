"""
Strength-vs-time curve. Plays a target agent against a fixed reference at
several time budgets and reports the win rate (+ 95% CI) at each point,
plus a crude ASCII plot.

A flat curve = the search saturates and more compute buys nothing. A steep
curve = compute matters; the headline single-budget number understates a
budget-sensitive change.

Example:
    python tools/strength_curve.py --target agent --ref variants.mcts_heavy \\
        -n 20 -j 4 --budgets 1 2 5 10
"""

import argparse
import math
import sys
import time

from _runner import run_pairings


def wilson_ci(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("-n", "--games-per-budget", type=int, default=20)
    parser.add_argument("-j", "--parallel", type=int, default=4)
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0],
                        help="per-agent time limits to test, seconds")
    args = parser.parse_args()

    results: list[tuple[float, int, int, int]] = []   # (budget, W, D, L)
    print(f"Target: {args.target}")
    print(f"Reference: {args.ref}")
    print(f"Budgets: {args.budgets}  n={args.games_per_budget}/budget")
    overall_start = time.monotonic()

    for budget in args.budgets:
        half = args.games_per_budget // 2
        pairings = [(args.target, args.ref)] * half
        pairings += [(args.ref, args.target)] * (args.games_per_budget - half)
        W = D = L = 0
        b_start = time.monotonic()
        for res in run_pairings(pairings, args.parallel, budget):
            if res.draw:
                D += 1
            elif res.winner == args.target:
                W += 1
            elif res.winner == args.ref:
                L += 1
        print(f"  budget={budget:5.1f}s  W={W} D={D} L={L}  ({time.monotonic()-b_start:.0f}s)", flush=True)
        results.append((budget, W, D, L))

    print()
    print("=" * 76)
    print(f"{'Budget(s)':>10} {'Win%':>7} {'Score':>7} {'95% CI':>14}  {'plot':<30}")
    for budget, W, D, L in results:
        n = W + D + L
        wr = W / n if n else 0.0
        score = (W + 0.5 * D) / n if n else 0.0
        lo, hi = wilson_ci(W, n)
        # Naive ASCII bar on score
        bar_len = 30
        filled = int(round(score * bar_len))
        bar = "█" * filled + "·" * (bar_len - filled)
        print(f"{budget:10.1f} {wr*100:6.1f}% {score:7.3f} [{lo:.2f},{hi:.2f}]  {bar}")
    print(f"Wall: {(time.monotonic()-overall_start)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
