# tools/bench.py

Round-robin head-to-head benchmark runner. Plays N games between two
referee-compatible modules with **balanced colours**, in parallel, and prints a
win-rate summary.

## Usage

```bash
python tools/bench.py --red <module> --blue <module> [options]
```

The two modules must each expose an `Agent` class — i.e. anything you can pass
to `python -m referee`. Examples: `agent`, `variants.ab1`, `variants.ab3`,
`variants/random_bot` (slash form also works).

### Options

| flag                  | default | meaning                                              |
|-----------------------|---------|------------------------------------------------------|
| `--red MODULE`        | —       | module played as Red in the first half               |
| `--blue MODULE`       | —       | module played as Blue in the first half              |
| `-n, --games N`       | 100     | total games (split evenly per side)                  |
| `-j, --parallel J`    | 4       | concurrent games (one process each)                  |
| `-t, --time-limit T`  | 60      | seconds of CPU per agent per game                    |

### Examples

```bash
# Full I3 exit test from the implementation plan (§7.4)
python tools/bench.py --red variants.ab3 --blue variants.ab2 -n 100 -j 4 -t 60

# Quick correctness check vs the random baseline
python tools/bench.py --red agent --blue variants.random_bot -n 20 -j 4 -t 10

# Champion vs frozen I3 once you've snapshotted ab_frozen_v3
python tools/bench.py --red agent --blue variants.ab_frozen_v3 -n 200 -j 8
```

## How it works

1. Builds N pairings: the first `N/2` use `--red` as Red, the remaining `N/2`
   swap them, so any colour bias cancels out in the aggregate.
2. Spawns up to `-j` games in parallel via `ProcessPoolExecutor`. Each worker
   shells out to `python -m referee -v 0 -t T <red_pkg> <blue_pkg>` and parses
   the trailing `result:` line.
3. Attributes each game to the **module** that won, not "player 1 / player 2"
   (those flip when colours swap), and prints a running tally.
4. At the end prints the aggregate score and win rate of the `--red` module.

## Output

Live progress lines look like:

```
[ 47/100]    variants.ab3 vs variants.ab2     -> variants.ab3 win  (38.2s)  running: variants.ab3=29 variants.ab2=15 D=3 F=0
```

The final block:

```
============================================================
Final: variants.ab3 67 | variants.ab2 30 | draws 3 | failures 0
variants.ab3 win rate: 67/100 = 67.0%
Wall time: 25.4 min
```

`failures` are games where the referee subprocess exited non-zero or the
`result:` line couldn't be parsed — usually a crash inside one of the agents.
Investigate before trusting the win-rate number.

## What it is not

- **Not an Elo system.** It only reports head-to-head win rate for a single
  pair. For ranking 3+ agents, run pair-wise benches and feed the results into
  an Elo calc separately (`tools/tournament.py`, planned).
- **Not a significance test.** 67/100 has a 95% CI of roughly ±9pp. For a real
  "agent A is stronger than agent B" claim, run ≥ 200 games or stop early
  using a sequential probability ratio test.
- **Not a profiler.** Use `cProfile` directly on the referee command if you
  want per-function timings.

## Limitations

- Cascade is deterministic given two fixed agents. Mirrored colours give some
  variation in the opening tree (Red places first, so Red and Blue see
  different positions), but two different `bench.py` runs of the same pair at
  the same parallelism produce identical results. Don't expect re-running to
  "average out" — it won't.
- The `-t` flag caps **CPU time per game**, not per move. Setting it well
  below the 180 s spec limit (e.g. 30 or 60) lets you run more games per hour
  but produces weaker play; results don't extrapolate linearly to full-budget
  strength. Quote `-t` whenever you cite a win rate.
- Parallelism is bounded by CPU cores. `-j 8` on a 4-core machine just slows
  every game down by ~2× — net throughput barely improves. Match `-j` to your
  physical core count.
