"""
Pool ratings using Glicko-2 over a round-robin of agents.

Each pairing is played as a small color-balanced mini-tournament. After all
games, ratings, deviations (RD), and volatilities are updated using the
standard Glicko-2 iteration (Glickman 2012).

Why Glicko-2 over plain Elo: it produces rating uncertainty (an RD band)
so you can distinguish "agent A is 50 Elo above agent B" from "agent A
might be 50 Elo above agent B but we've only played 8 games". Avoids the
trap of false confidence in small samples.

Example:
    python tools/ratings.py \\
        --agents agent variants.mcts_final variants.mcts_heavy variants.ab3 variants.greedy \\
        -n 6 -t 5 -j 4

`-n` is games per *ordered pair* (so each pair plays 2n with colors balanced).
"""

import argparse
import itertools
import math
import sys
import time

from _runner import run_pairings


# Glicko-2 constants
GLICKO_SCALE = 173.7178
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
TAU = 0.5  # system constant; smaller = slower rating change


def g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))


def E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-g(phi_j) * (mu - mu_j)))


def glicko2_update(rating: float, rd: float, vol: float, opponents):
    """
    opponents: list of (opp_rating, opp_rd, score) tuples.
    Returns (new_rating, new_rd, new_vol).
    """
    if not opponents:
        # Inactivity: RD increases (Step 8 with empty period).
        phi = rd / GLICKO_SCALE
        new_phi = math.sqrt(phi * phi + vol * vol)
        return rating, new_phi * GLICKO_SCALE, vol

    mu = (rating - DEFAULT_RATING) / GLICKO_SCALE
    phi = rd / GLICKO_SCALE

    v_inv = 0.0
    delta_num = 0.0
    for opp_r, opp_rd, s in opponents:
        mu_j = (opp_r - DEFAULT_RATING) / GLICKO_SCALE
        phi_j = opp_rd / GLICKO_SCALE
        gj = g(phi_j)
        Ej = E(mu, mu_j, phi_j)
        v_inv += gj * gj * Ej * (1.0 - Ej)
        delta_num += gj * (s - Ej)
    v = 1.0 / v_inv if v_inv > 0 else float("inf")
    delta = v * delta_num

    a = math.log(vol * vol)
    eps = 1e-6

    def f(x):
        ex = math.exp(x)
        return ex * (delta * delta - phi * phi - v - ex) / (2 * (phi * phi + v + ex) ** 2) - (x - a) / (TAU * TAU)

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
            if k > 100:
                break
        B = a - k * TAU

    fA, fB = f(A), f(B)
    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2
        B, fB = C, fC

    new_vol = math.exp(A / 2)
    phi_star = math.sqrt(phi * phi + new_vol * new_vol)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = mu + new_phi * new_phi * delta_num

    return (new_mu * GLICKO_SCALE + DEFAULT_RATING,
            new_phi * GLICKO_SCALE,
            new_vol)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", required=True, help="agent module paths to rate")
    parser.add_argument("-n", "--games-per-pair", type=int, default=4,
                        help="games per ordered pair; doubled for color balance")
    parser.add_argument("-t", "--time-limit", type=float, default=5.0)
    parser.add_argument("-j", "--parallel", type=int, default=4)
    args = parser.parse_args()

    agents = args.agents
    pairings: list[tuple[str, str]] = []
    for a, b in itertools.combinations(agents, 2):
        for _ in range(args.games_per_pair):
            pairings.append((a, b))
            pairings.append((b, a))

    print(f"Pool: {len(agents)} agents, {len(pairings)} games, t={args.time_limit}s, j={args.parallel}")
    overall_start = time.monotonic()

    # Per-agent log of (opp, score) pairs for Glicko-2 update.
    opp_log: dict[str, list[tuple[str, float]]] = {a: [] for a in agents}
    score_matrix: dict[tuple[str, str], list[float]] = {}

    completed = 0
    for res in run_pairings(pairings, args.parallel, args.time_limit):
        completed += 1
        sa = res.outcome_for(res.red)
        sb = 1.0 - sa if not res.draw else 0.5
        opp_log[res.red].append((res.blue, sa))
        opp_log[res.blue].append((res.red, sb))
        score_matrix.setdefault((res.red, res.blue), []).append(sa)
        if completed % 5 == 0 or completed == len(pairings):
            print(f"  [{completed}/{len(pairings)}] done", flush=True)

    # Single Glicko-2 rating period (everyone updated once from accumulated games).
    ratings = {a: (DEFAULT_RATING, DEFAULT_RD, DEFAULT_VOL) for a in agents}
    new_ratings = {}
    for a in agents:
        opps = [(ratings[opp][0], ratings[opp][1], s) for opp, s in opp_log[a]]
        new_ratings[a] = glicko2_update(*ratings[a], opps)
    ratings = new_ratings

    elapsed = time.monotonic() - overall_start
    print()
    print("=" * 78)
    print(f"{'Agent':<40} {'Rating':>8} {'± 95% RD':>10} {'Games':>7}")
    print("-" * 78)
    for a in sorted(agents, key=lambda x: -ratings[x][0]):
        r, rd, _vol = ratings[a]
        print(f"{a:<40} {r:8.0f} {1.96*rd:10.0f} {len(opp_log[a]):7d}")
    print()
    print("Pairwise score matrix (row = A, cell = A's score vs B):")
    print(f"{'':<25} " + " ".join(f"{b[-12:]:>13}" for b in agents))
    for a in agents:
        row = []
        for b in agents:
            if a == b:
                row.append(f"{'—':>13}")
            else:
                scores = score_matrix.get((a, b), []) + [1 - s for s in score_matrix.get((b, a), [])]
                if scores:
                    row.append(f"{sum(scores)/len(scores):>13.2f}")
                else:
                    row.append(f"{'n/a':>13}")
        print(f"{a[-25:]:<25} " + " ".join(row))
    print()
    print(f"Wall time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
