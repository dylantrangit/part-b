"""
Round-robin benchmark runner: pit any two referee-compatible modules against
each other for N games with balanced colours, run J in parallel.

Examples:
    python tools/bench.py --red agent --blue variants.ab1 -n 100 -j 4
    python tools/bench.py --red variants.ab2 --blue variants.ab1 -n 50 -t 30
"""

import argparse
import concurrent.futures
import re
import subprocess
import sys
import time

RESULT_RE = re.compile(r"result:\s*(player\s+\d|draw)\s*(?:\[([^\]]+)\])?", re.IGNORECASE)


def play_one(red, blue, game_idx, time_limit):
    cmd = ["python", "-m", "referee", "-v", "0", "-t", str(time_limit), red, blue]
    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - start
    output = (proc.stdout or "") + (proc.stderr or "")

    winner_module = None
    is_draw = False
    for line in reversed(output.splitlines()):
        m = RESULT_RE.search(line)
        if m:
            tag = m.group(1).lower().strip()
            if "draw" in tag:
                is_draw = True
            else:
                bracket = m.group(2) or ""
                if ":" in bracket:
                    winner_module = bracket.split(":", 1)[0]
                elif "1" in tag:
                    winner_module = red
                elif "2" in tag:
                    winner_module = blue
            break

    return {
        "idx": game_idx,
        "red": red,
        "blue": blue,
        "winner": winner_module,
        "draw": is_draw,
        "elapsed": elapsed,
        "rc": proc.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--red", required=True, help="module path for the 'red' agent (e.g. agent, variants.ab1)")
    parser.add_argument("--blue", required=True, help="module path for the 'blue' agent")
    parser.add_argument("-n", "--games", type=int, default=100, help="total games (split evenly per side)")
    parser.add_argument("-j", "--parallel", type=int, default=4, help="concurrent games")
    parser.add_argument("-t", "--time-limit", type=int, default=60, help="seconds per agent per game")
    args = parser.parse_args()

    red, blue = args.red, args.blue
    if red == blue:
        print("warning: red and blue are the same module — both wins are attributed to that module")
    half = args.games // 2

    pairings = []
    for i in range(half):
        pairings.append((red, blue, i))
    for i in range(args.games - half):
        pairings.append((blue, red, half + i))

    red_wins = 0
    blue_wins = 0
    draws = 0
    failures = 0

    print(f"Running {len(pairings)} games at {args.parallel}-way parallelism, -t {args.time_limit}s/agent.")
    print(f"Red module:  {red}")
    print(f"Blue module: {blue}")
    overall_start = time.monotonic()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(play_one, r, b, i, args.time_limit) for r, b, i in pairings]
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            res = fut.result()
            if res["draw"]:
                draws += 1
                outcome = "draw"
            elif res["winner"] == red:
                red_wins += 1
                outcome = f"{red} win"
            elif res["winner"] == blue:
                blue_wins += 1
                outcome = f"{blue} win"
            else:
                failures += 1
                outcome = f"FAILED rc={res['rc']}"
            print(
                f"[{completed:3d}/{len(pairings)}] "
                f"{res['red']:>16} vs {res['blue']:<16} -> {outcome:<24} "
                f"({res['elapsed']:.1f}s)  "
                f"running: {red}={red_wins} {blue}={blue_wins} D={draws} F={failures}",
                flush=True,
            )

    overall_elapsed = time.monotonic() - overall_start
    total = red_wins + blue_wins + draws + failures
    print()
    print("=" * 60)
    print(f"Final: {red} {red_wins} | {blue} {blue_wins} | draws {draws} | failures {failures}")
    if total:
        print(f"{red} win rate: {red_wins}/{total} = {100 * red_wins / total:.1f}%")
    print(f"Wall time: {overall_elapsed/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
