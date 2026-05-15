# tools/

Evaluation toolkit for the Cascade agent. All tools are read-only with respect
to `agent/` — none of them modify the shipping code. They subprocess
`python -m referee` for game-running (so an agent crash is isolated) and
import `agent.core.*` / `agent.search.*` directly only when they need to drive
the search on a specific position.

Run everything from the project root. Tools that import `agent.*` insert the
project root onto `sys.path` themselves, so `python tools/<x>.py` works.

| Tool                  | What it answers                                              |
|-----------------------|--------------------------------------------------------------|
| `bench.py`            | Head-to-head win rate (with optional `--paired` variance reduction). |
| `sprt.py`             | Is A *significantly* stronger than B? Sequential test, stops early. |
| `ratings.py`          | Pool ranking with confidence bands via Glicko-2.             |
| `strength_curve.py`   | How does win rate change as time budget changes?             |
| `pv_stability.py`     | Per-depth principal-variation stability for the α–β search.  |
| `tactical_runner.py`  | % of hand-built tactical puzzles solved.                     |
| `calibration.py`      | Does the eval's predicted P(win) match empirical win rate?   |

---

## bench.py — basic head-to-head

**Why it's here.** Every "X beats Y at Z%" claim in the report — iteration
exit tests (e.g. I3 ≥ 65% vs I2), the final agent's score against the frozen
α–β baseline, the placement-policy ablation — ultimately comes from this
tool. It is the smallest unit of evidence the project produces.

**What it does.** Plays `N` games of Cascade between two agent modules by
subprocess-spawning `python -m referee <red> <blue>`. Half the games are
played with each agent as Red (Red moves first in Cascade, which is a real
advantage, so colour-balancing is mandatory for honest numbers). Reports the
win/loss/draw counts and a win rate.

The `--paired` mode reduces noise when the two agents are close in strength.
In plain mode, each of the `N` games is independent. In paired mode, the
tool generates `N/2` "ordinals" and plays each ordinal twice — once with A as
Red, once with B as Red — using the same starting conditions. The two
results are paired up, so any first-move advantage cancels exactly inside
each pair rather than only on average across `N`. This is the same trick
used in paired t-tests and in computer-chess testing.

```bash
# Plain mode: N games, half each as Red.
python tools/bench.py --red agent --blue variants.mcts_heavy -n 100 -j 4 -t 5

# Paired mode: each ordinal played both ways (A-as-Red, then B-as-Red).
python tools/bench.py --red agent --blue variants.mcts_heavy -n 100 -j 4 -t 5 --paired
```

## sprt.py — sequential probability ratio test

**Why it's here.** When deciding whether to *promote* a new iteration to be
the shipping agent (e.g. swap I5 for I6 in `agent/program.py`), running a
fixed-N bench is wasteful: if the new variant is clearly much stronger or
clearly much weaker, you can stop after 40 games; if they're nearly
identical, no fixed N will resolve the question and you're better off not
flipping the coin. SPRT answers the promotion question with the *minimum*
games needed, and refuses to answer at all when the data doesn't support a
conclusion.

**What it does.** SPRT (Wald, 1945) is a hypothesis test that consumes
samples one at a time and decides as soon as the evidence crosses a
threshold. It tests two hypotheses:

- **H0:** "A is no stronger than B" — pinned to a strength gap of `elo0`
  (default 0, i.e. equal strength).
- **H1:** "A is meaningfully stronger than B" — pinned to a strength gap of
  `elo1` (default 15 Elo, the smallest gap we care to detect).

After each game the test updates a *likelihood ratio* — informally, "how
many times more probable are the games we've seen if H1 is true vs if H0 is
true". The log of that ratio is just running sum of per-game contributions,
which is cheap. When the log-likelihood ratio rises above an upper bound
(set by the user's allowed false-accept rate `alpha`, default 5%), accept
H1: ship the new variant. When it falls below a lower bound (false-reject
rate `beta`, default 5%), accept H0: keep the old one. Until then, keep
playing. `--max-games` caps the run so an inconclusive test doesn't loop
forever.

The headline property: for the same error rates, SPRT typically needs
~50% fewer games than a fixed-N test. Computer-chess engine development
adopted it widely for the same reason (Fishtest / Stockfish).

References: Wald (1945), *Sequential Tests of Statistical Hypotheses*;
Stockfish's `fishtest` framework uses the same Elo-based formulation.

```bash
python tools/sprt.py --A agent --B variants.mcts_heavy \
    -t 5 -j 4 --elo0 0 --elo1 15 --alpha 0.05 --beta 0.05 --max-games 400
```

## ratings.py — pool ratings with confidence bands

**Why it's here.** The performance-evaluation section of the report needs a
single picture of how every iteration stacks up — greedy, α–β at I1/I2/I3,
MCTS at I4/I5/I6, the random baseline. A pile of pairwise win rates is hard
to read; a sorted ranking with uncertainty bars is one figure that tells the
whole story.

**What it does.** Plays a colour-balanced round-robin: every unordered pair
of agents plays `2n` games (`n` with one agent as Red, `n` with the other),
then assigns each agent a numerical rating reflecting its overall strength.

Two pieces of background:

- **Elo (Arpad Elo, 1960s)** is the original rating system from chess. Each
  player has one number; the *gap* between two ratings predicts how often
  the stronger player wins. A 200-point gap means the stronger player is
  expected to win about 76% of games; 400 points means ~91%. After each
  game, both ratings shift by a fixed step size based on whether the result
  was surprising. Elo's flaw for our purposes: it has no notion of
  uncertainty — a player who's played 4 games and one who's played 400 have
  ratings that look equally trustworthy, even though the first is mostly
  noise.

- **Glicko-2 (Glickman, 2012)** fixes that. Each agent gets three numbers
  instead of one: a rating, a *rating deviation* (RD) measuring how
  uncertain we are about that rating, and a volatility measuring how
  erratic the agent's results are. The RD shrinks as more games accumulate;
  rating updates are larger when RD is high (uncertain players move fast)
  and smaller when RD is low (well-established players move slowly). Most
  importantly, the RD gives a *confidence band*: "A is rated 1650, ± 60
  with 95% confidence" is a real statement; "A is 50 Elo above B" without a
  band is not.

The tool reports each agent's rating, its 95% band (`± 1.96 × RD`), and a
pairwise score matrix. Bands that overlap mean the apparent ordering of two
agents isn't statistically distinguishable yet — run more games.

References: Elo (1978), *The Rating of Chessplayers, Past and Present*;
Glickman (2012), *Example of the Glicko-2 system* (the canonical
implementation reference, which this file's `glicko2_update` follows).

```bash
python tools/ratings.py \
    --agents agent variants.mcts_final variants.mcts_heavy variants.ab3 variants.greedy \
    -n 4 -t 5 -j 4
```

`-n` is games per *ordered pair*; each pair plays `2n` (colour balanced).

## strength_curve.py — win rate vs time budget

**Why it's here.** The report needs to justify the time-management policy in
`agent/core/time_budget.py` and to argue that the MCTS search is actually
*using* extra compute productively. The plot it produces also lets the
reader judge whether the 5 s/move local benchmark is representative of how
the agent will behave at Gradescope's larger budget.

**What it does.** Fixes a target agent and a reference opponent, then
replays the same head-to-head match at a sweep of time budgets (e.g. 1 s,
2 s, 5 s, 10 s per move) and prints the win rate at each. The shape of the
resulting curve is the diagnostic:

- A **steep upward** curve means the target gets meaningfully stronger with
  more thinking time — typical of MCTS, which exploits extra simulations.
- A **flat** curve means the search has saturated — either the target has
  hit some structural ceiling (eval is the limit, not search) or the
  position is too easy / too hard for compute to matter.
- A **downward** curve indicates a bug — the search is making *worse*
  decisions with more time (e.g. instability in the eval, or a time-budget
  policy that misallocates the extra budget).

The same opponent at the same budget is used on both sides of each
comparison, so the curve measures the *target's* response to compute, not a
joint effect.

```bash
python tools/strength_curve.py --target agent --ref variants.mcts_heavy \
    -n 20 -j 4 --budgets 1 2 5 10
```

## pv_stability.py — principal-variation stability

**Why it's here.** The α–β benchmark agent (I3 in the implementation plan)
uses iterative deepening: it searches depth 1, then depth 2, then 3, etc.,
keeping the best move from the deepest *completed* search. If the eval is
noisy or the move ordering is poor, the best move can flip between depths,
which means the agent is essentially guessing. This tool measures that
flipping rate, and the report uses it to argue the eval is well-behaved.

**What it does.** A "principal variation" (PV) in α–β search is the sequence
of moves the search currently believes both sides will play if they play
optimally; the first move of the PV is what the agent would play *right
now* if forced to commit. The tool drives the α–β backend
(`agent.search.pvs._pvs`) on a sampled mid-game position, running fresh
searches at depths 1 through `D`. At each depth it records the best move
and its evaluation score, then computes two metrics:

- **Stability**: the fraction of depths whose chosen best move agrees with
  the final (deepest) search's choice. A well-tuned search reaches ~80–90%
  stability — most depths already agree on the right move, with only the
  shallowest disagreeing.
- **Lock-in depth**: the smallest depth from which the best move never
  changes again. Lower is better — it means the agent can commit early and
  the extra plies of search are confirming, not flipping, the decision.

This is α–β only by construction; MCTS doesn't have a meaningful per-depth
PV (it has a tree of visit counts, not a depth ladder), so use
`tactical_runner.py` for MCTS quality checks instead.

```bash
python tools/pv_stability.py --max-depth 7 -t 30 --positions 5
```

## tactical_runner.py — hand-built puzzle suite

**Why it's here.** Round-robin win rates show that one agent beats another,
but not *why*. The report's strongest evidence-of-technique is concrete:
"I3 misses this forced win in 3 plies because of the horizon effect; I6
solves it because MCTS-Solver propagates a proven loss for the opponent up
the tree". To make that kind of claim, the project needs a curated set of
positions where the correct move is known.

**What it does.** Loads positions defined in `tests/tactical/positions.py`,
each labelled with the expected best move (or, for blunder-avoidance tests,
a set of forbidden moves) and a difficulty tag. For each position, the
runner builds the board directly via `GameState.set_cell` (bypassing the
placement phase), calls the chosen search backend at a fixed time budget,
and records whether the agent found the expected move.

Backends: `pvs`, `ab`, `mcts` / `mcts_heavy`, `mcts_final`.

The output is a per-tag pass rate. Two uses in the report:

1. **Cross-iteration ablation** — running the same suite against I1, I2,
   I3, I5, I6 produces a "puzzles solved" curve that grows monotonically as
   each technique is added (quiescence, RAVE, MCTS-Solver).
2. **Qualitative anecdote** — picking 1–2 puzzles where I6 succeeds and I3
   fails, and walking through *why* in the report, is more persuasive than
   any aggregate win rate.

```bash
python tools/tactical_runner.py --backend pvs -t 2
python tools/tactical_runner.py --backend mcts_heavy -t 2
python tools/tactical_runner.py --backend mcts_final -t 2 --tags eat win-elimination
```

## calibration.py — is the eval well-calibrated?

**Why it's here.** Two specific decisions depend on the eval producing
sensible numbers, not just sensible *orderings*: the `tanh(eval / scale)`
mapping used to convert α–β scores into MCTS rewards (see §9.1 and §10.1d
of the implementation plan), and the `scale ≈ 500` constant inside it.
Without calibration data, that constant is pulled from thin air. With it,
the report can defend the chosen scale as the one that matches the eval's
actual predictive accuracy.

**What it does.** A calibrated probabilistic predictor is one where, of all
the times it says "70% chance of winning", the side it favoured really did
win 70% of the time. The tool measures this for our eval.

It plays N games between two agents, samples positions along each game,
and for every sampled position:

1. Computes `evaluate(state)` — a signed score in centipawn-ish units.
2. Maps it through a sigmoid `1 / (1 + exp(-eval / scale))` to a predicted
   probability that Red wins from that position.
3. Records the actual outcome of the game that position came from.

Positions are then bucketed by predicted probability (e.g. 0.0–0.1, 0.1–0.2,
…, 0.9–1.0). For each bucket, the tool compares the bucket's *average
predicted* P(red wins) to the *empirical* fraction of games in that bucket
that Red actually won. A well-calibrated eval has these two numbers close
in every bucket (small RMSE). An over-confident eval (scale too small)
predicts 95% when the real rate is 70%; an under-confident eval (scale too
large) predicts 55% when the real rate is 90%. Re-running with different
`--scale` values is how the right constant gets pinned down.

Reference: Niculescu-Mizil & Caruana (2005), *Predicting Good Probabilities
With Supervised Learning* — same diagnostic (reliability diagrams), applied
to ML classifiers rather than game evals.

```bash
python tools/calibration.py --A agent --B variants.mcts_heavy \
    -n 30 -j 4 -t 5 --scale 300
```

---

## Limitations

- **Compute time.** A full SPRT or rating pool can run for hours. Start with
  small `-n` / `--max-games` to confirm the tool fits your workflow, then
  scale up.
- **Calibration scale is hand-set.** The right `--scale` for our eval is
  unknown until you've run calibration once and looked at the spread.
- **PV stability is α–β only.** MCTS doesn't have a meaningful per-depth PV
  by construction. Use `tactical_runner.py` for MCTS quality checks instead.
- **Subprocess overhead.** Each game spawns a fresh Python interpreter. At
  `-t 1` and below, the import-time dominates and win rates are misleading.
  Use `-t ≥ 3` for trustworthy numbers.
