"""
Sequential Probability Ratio Test (SPRT) for two-agent strength comparison.

Runs color-paired games between A and B until the log-likelihood ratio
between H0 (A is no stronger than `elo0`) and H1 (A is at least `elo1` Elo
stronger) crosses a decision threshold derived from (alpha, beta) Type I/II
error rates.

This is the protocol used by chess engine devs (Fishtest, OpenBench) to
get tight strength comparisons from small game counts. Pairs each game's
mirror so every (Red, Blue) is matched by a (Blue, Red).

Example:
    python tools/sprt.py --A agent --B variants.mcts_heavy_no_opening_pvs \\
        -t 5 -j 4 --elo0 0 --elo1 15 --alpha 0.05 --beta 0.05 --max-games 400

References:
    - Wald 1945, "Sequential Tests of Statistical Hypotheses"
    - https://github.com/glinscott/fishtest/wiki/Fishtest-SPRT-Calculations
"""

import argparse
import math
import sys
import time

from _runner import run_pairings


def elo_to_score(elo: float) -> float:
    """Logistic Elo to expected score in [0, 1]."""
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def llr_normalized(W: int, D: int, L: int, elo0: float, elo1: float) -> float:
    """
    Generalized log-likelihood ratio for a trinomial (W, D, L) outcome
    distribution, comparing H0: elo=elo0 vs H1: elo=elo1. Follows the
    "pentanomial-free" trinomial formulation in Michel Van den Bergh's
    SPRT note used by chess testing frameworks.
    """
    N = W + D + L
    if N == 0:
        return 0.0
    # Empirical probabilities.
    pw, pd, pl = W / N, D / N, L / N
    # MLE under each hypothesis: fit only drawelo / win prob. Use the
    # simpler 2-parameter trinomial: estimate p_w + p_d/2 as score and
    # variance from sample. This matches Fishtest's `LL` for trinomial.
    score = pw + 0.5 * pd
    # variance of a single game's score around the mean:
    var = pw * (1 - score) ** 2 + pd * (0.5 - score) ** 2 + pl * (0 - score) ** 2
    if var <= 0:
        return 0.0
    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)
    return (s1 - s0) * (2.0 * N * score - N * (s0 + s1)) / (2.0 * N * var)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--A", required=True, help="candidate module (the one being tested)")
    parser.add_argument("--B", required=True, help="reference module")
    parser.add_argument("-t", "--time-limit", type=float, default=5.0, help="seconds per agent per game")
    parser.add_argument("-j", "--parallel", type=int, default=4)
    parser.add_argument("--elo0", type=float, default=0.0, help="H0: A is at most this many Elo above B")
    parser.add_argument("--elo1", type=float, default=15.0, help="H1: A is at least this many Elo above B")
    parser.add_argument("--alpha", type=float, default=0.05, help="Type I error: prob of accepting H1 when H0 true")
    parser.add_argument("--beta", type=float, default=0.05, help="Type II error: prob of accepting H0 when H1 true")
    parser.add_argument("--max-games", type=int, default=400, help="hard cap; stop with 'inconclusive' if hit")
    parser.add_argument("--batch", type=int, default=8, help="games per parallel batch (must be even for pairing)")
    args = parser.parse_args()

    if args.batch % 2:
        args.batch += 1

    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)
    print(f"SPRT bounds: lower={lower:.3f} (accept H0), upper={upper:.3f} (accept H1)")
    print(f"A={args.A}  B={args.B}  H0: <={args.elo0} Elo  H1: >={args.elo1} Elo  alpha={args.alpha} beta={args.beta}")

    W = D = L = 0
    games = 0
    start = time.monotonic()
    decision = "inconclusive"

    while games < args.max_games:
        remaining = args.max_games - games
        batch_size = min(args.batch, remaining)
        # Pair the batch: half A-as-red, half B-as-red.
        half = batch_size // 2
        pairings = [(args.A, args.B)] * half + [(args.B, args.A)] * (batch_size - half)

        for res in run_pairings(pairings, args.parallel, args.time_limit):
            games += 1
            if res.draw:
                D += 1
                tag = "D"
            elif res.winner == args.A:
                W += 1
                tag = "W"
            elif res.winner == args.B:
                L += 1
                tag = "L"
            else:
                tag = "F"
                # failures don't update LLR
                print(f"  [game {games}] FAILED rc={res.rc}; not counted")
                continue

            llr = llr_normalized(W, D, L, args.elo0, args.elo1)
            score = (W + 0.5 * D) / max(1, W + D + L)
            print(f"  [game {games}] {tag} W={W} D={D} L={L}  score={score:.3f}  LLR={llr:+.3f}",
                  flush=True)

            if llr >= upper:
                decision = "ACCEPT H1 (A is stronger)"
                break
            if llr <= lower:
                decision = "ACCEPT H0 (A is not stronger)"
                break

        if decision != "inconclusive":
            break

    elapsed = time.monotonic() - start
    print()
    print("=" * 60)
    print(f"Decision: {decision}")
    print(f"Games: W={W} D={D} L={L} ({games} total)")
    if games:
        score = (W + 0.5 * D) / games
        print(f"Score: {score:.3f}  (≈ {score_to_elo(score):+.1f} Elo, naive)")
    print(f"Wall time: {elapsed/60:.1f} min")


def score_to_elo(score: float) -> float:
    """Inverse of elo_to_score; clamp to avoid log(0)."""
    s = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


if __name__ == "__main__":
    sys.exit(main())
