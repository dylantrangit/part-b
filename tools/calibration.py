"""
Evaluation-function calibration.

Plays N games between two agents (any pair — self-play also fine), then
for every position visited along the way, records (eval, eventual_outcome).
Buckets positions by their predicted win probability (sigmoid of eval),
and reports the empirical win rate within each bucket. A well-calibrated
eval has empirical ≈ predicted across buckets.

Mis-calibration shows up as off-diagonal mass in the table:
    - "eval says +500 but red only wins 50% of the time" → the eval is
      over-confident and the weights probably weight the wrong feature.
    - "eval ranges only over ±100 but games end with token diffs of ±12" →
      the eval is under-confident; weights too small.

Example:
    python tools/calibration.py --A agent --B variants.mcts_heavy \\
        -n 30 -j 4 -t 5 --scale 300
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from referee.game import (
    Coord, Direction,
    PlaceAction, MoveAction, EatAction, CascadeAction,
)

from _runner import run_pairings
from agent.core.board import GameState
from agent.core.eval import evaluate


_DIR_MAP = {
    "↑": Direction.Up, "Up": Direction.Up,
    "↓": Direction.Down, "Down": Direction.Down,
    "←": Direction.Left, "Left": Direction.Left,
    "→": Direction.Right, "Right": Direction.Right,
}


def parse_action(kind: str, args: str):
    if kind == "PLACE":
        r, c = args.split("-")
        return PlaceAction(Coord(int(r), int(c)))
    coord_part, dir_part = args.split(",", 1)
    r, c = coord_part.strip().split("-")
    coord = Coord(int(r), int(c))
    direction = _DIR_MAP[dir_part.strip().lstrip("[").rstrip("]")]
    if kind == "MOVE":
        return MoveAction(coord, direction)
    if kind == "EAT":
        return EatAction(coord, direction)
    if kind == "CASCADE":
        return CascadeAction(coord, direction)
    raise ValueError(f"unknown action {kind!r}")


def sigmoid(x: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / scale))


def replay_samples(transcript: list[str], stride: int):
    """Yield (red_eval, play_ply) at every `stride`th play-phase ply."""
    state = GameState()
    yield evaluate(state), 0
    last_yielded = 0
    for entry in transcript:
        side, kind, args = entry.split(" ", 2)
        action = parse_action(kind, args)
        state.apply(action)
        if state.play_ply > 0 and state.play_ply - last_yielded >= stride:
            yield evaluate(state), state.play_ply
            last_yielded = state.play_ply


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--A", required=True)
    parser.add_argument("--B", required=True)
    parser.add_argument("-n", "--games", type=int, default=30)
    parser.add_argument("-j", "--parallel", type=int, default=4)
    parser.add_argument("-t", "--time-limit", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=300.0,
                        help="sigmoid scale: maps eval to predicted P(red wins)")
    parser.add_argument("--stride", type=int, default=4,
                        help="sample every `stride` play plies")
    parser.add_argument("--bins", type=int, default=10,
                        help="number of predicted-P bins")
    args = parser.parse_args()

    half = args.games // 2
    pairings = [(args.A, args.B)] * half + [(args.B, args.A)] * (args.games - half)

    # Bucketed (predicted_p_bin, count_red_wins, total_count)
    bins = args.bins
    win_counts = [0] * bins
    tot_counts = [0] * bins
    total_samples = 0

    start = time.monotonic()
    games_done = 0
    for res in run_pairings(pairings, args.parallel, args.time_limit, verbose=1):
        games_done += 1
        # Red-result ground truth for this game.
        if res.draw:
            red_outcome = 0.5
        elif res.winner == res.red:
            red_outcome = 1.0
        else:
            red_outcome = 0.0
        try:
            for ev, _ply in replay_samples(res.transcript, args.stride):
                p = sigmoid(ev, args.scale)
                b = min(bins - 1, int(p * bins))
                win_counts[b] += red_outcome
                tot_counts[b] += 1
                total_samples += 1
        except Exception as e:
            print(f"  [game {games_done}] replay error ({type(e).__name__}); skipping")
            continue
        print(f"  [game {games_done}/{len(pairings)}] samples={total_samples}", flush=True)

    elapsed = time.monotonic() - start
    print()
    print("=" * 72)
    print(f"Pairs A={args.A}  B={args.B}   games={games_done}  samples={total_samples}")
    print(f"Sigmoid scale: eval/{args.scale} -> P(red wins)")
    print()
    print(f"{'bin':>5} {'pred-range':>14} {'avg pred':>9} {'empirical':>11} {'count':>7}  plot")
    overall_mse = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        n = tot_counts[i]
        if n == 0:
            print(f"  {i:3d} [{lo:.2f},{hi:.2f}]  {'—':>9} {'—':>11} {n:7d}  (empty)")
            continue
        avg_pred = (lo + hi) / 2
        emp = win_counts[i] / n
        diff = abs(emp - avg_pred)
        overall_mse += (emp - avg_pred) ** 2 * n
        bar = "█" * int(round(emp * 30)) + "·" * (30 - int(round(emp * 30)))
        marker = " <" + ("=" * int(min(10, diff * 20))) if diff > 0.05 else ""
        print(f"  {i:3d} [{lo:.2f},{hi:.2f}]  {avg_pred:9.2f} {emp:11.3f} {n:7d}  {bar}{marker}")

    if total_samples:
        rmse = math.sqrt(overall_mse / total_samples)
        print(f"\nWeighted RMSE (predicted vs empirical, per sample): {rmse:.3f}")
        print(f"  (0 = perfect, 0.5 = worst; rough rule of thumb: <0.15 is decent.)")
    print(f"Wall: {elapsed/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
