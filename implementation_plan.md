# Implementation Plan — Cascade Agent (α–β → Sophisticated MCTS)

The first shipped agent is already a working α–β player. From there, the search is progressively
strengthened, then the architecture is pivoted to Monte Carlo Tree Search and hardened
iteration by iteration until it reaches a sophisticated MCTS variant (MCTS-Solver +
PUCT priors + tree reuse + α–β hybrid).

The shared foundations (state representation, legal-action generator, Zobrist hashing,
terminal detection, placement-phase policy, evaluation function, testing harness) are
described once up-front so that each iteration section can focus on *what the search
changes*.

---

## 1. Ground truth (unchanged from spec)

- **Interface**: `agent/program.py :: Agent` with `__init__(color, **referee)`,
  `action(**referee) -> Action`, `update(color, action, **referee)`. Do not modify
  `referee/`.
- **Budget**: 180 s CPU, 250 MB RAM per agent per game. No disk I/O, no network, no
  threads. Python 3.12 + stdlib + NumPy only.
- **Actions**: `PlaceAction(coord)`, `MoveAction(coord, dir)`, `EatAction(coord, dir)`,
  `CascadeAction(coord, dir)`. Cardinal directions only.
- **End conditions** (precedence order): elimination → threefold repetition (draw) →
  stalemate (draw) → 300-turn limit (more tokens wins, tie = draw).
- **Marks target**: all 11 performance marks plus 10+/11 technique marks by shipping a
  sophisticated MCTS and documenting the α–β progression as the benchmark agent in the
  report (the spec explicitly recommends having a "benchmark agent" before MCTS).

---

## 2. Module layout

```
agent/
├── __init__.py          # unchanged — re-exports Agent
├── program.py           # Agent class; dispatches to the current search backend
├── board.py             # internal state + legal action generator + apply/undo
├── zobrist.py           # 64-bit Zobrist keys for TT and repetition
├── eval.py              # evaluation function + incremental features
├── ordering.py          # move ordering: TT move, MVV-LVA, killers, history, static
├── search.py            # α–β backend (I1–I3)
├── tt.py                # transposition table (bounded dict)
├── mcts.py              # MCTS backend (I4–I6)
├── policy.py            # hand-crafted prior policy for PUCT (I6)
├── placement.py         # opening / placement-phase policy
└── time_budget.py       # wall-clock and per-move time accounting
```

`program.py` exposes a single `SEARCH_BACKEND` constant (`"ab"` or `"mcts"`) so that
self-play comparisons can be run without forking the agent module — see §11.

---

## 3. Shared foundations (build these once, first)

These are not an "iteration 0" shipping milestone — they are the substrate that I1
builds on before the first submission.

### 3.1 State representation — `board.py`

- **Grid**: `np.ndarray` shape `(8, 8)` of `int8`. `0` empty, `+h` red stack of height
  `h`, `-h` blue stack of height `h`. Signed encoding makes negamax sign-flipping
  trivial and avoids branches in the eval.
- **Per-colour piece lists**: `list[int]` of flat `r*8+c` indices. Maintained
  incrementally so move generation iterates only occupied cells (never scans 64).
- **Incrementally maintained scalars**: `red_tokens`, `blue_tokens`, `red_stacks`,
  `blue_stacks`, `placement_count`, `turn_color`, `play_ply`.
- **Zobrist hash**: `uint64`. A table `Z[flat_idx, colour, height_bucket]` is built
  once in `zobrist.py` (heights bucketed up to 12 — plenty, since merges are rare and
  the total tokens per side is bounded). Side-to-move key is XORed in on every turn
  switch.
- **Play-phase history**: `dict[uint64, int]` for O(1) threefold-repetition checks.
  Entries from the placement phase are *not* stored (per spec).
- **Undo stack**: each apply pushes a tiny tuple `(touched_cells, prev_values,
  hash_delta, scalar_deltas)`. Cascades touch at most ~8 cells; the tuple stays
  tight. All mutations are in place — no state cloning in the hot path.

This is the only board implementation the agent uses during search. The referee's
`Board` stays untouched and is only consulted in tests for cross-checking.

### 3.2 Legal action generator — `board.py`

One generator per phase, yielding actions in a weak pre-order (TT / heuristics reorder
later).

- **Placement**: iterate empty cells; reject cells adjacent to any opponent stack
  (this restriction kicks in *after* the very first placement of the game).
- **Play**: iterate *our* piece list. For each stack at `(r, c)` with height `h` and
  each of four cardinal dirs:
  - Destination out of bounds → only CASCADE may still be legal.
  - Destination empty → `MoveAction` (relocate).
  - Destination friendly → `MoveAction` (merge).
  - Destination enemy and `h ≥ enemy.h` → `EatAction`.
  - If `h ≥ 2`, emit `CascadeAction` regardless of what lies ahead (cascades are
    always legal once `h ≥ 2`, even if they self-eliminate).
- **Stalemate**: if the generator is empty for the side to move, the position is a
  draw. Cross-checked against `referee/game/board.py:306`.

Expected branching factor: ~30–80 play-phase moves in the midgame.

### 3.3 Terminal detection

At every search node, before any expansion:

1. Elimination → immediate terminal (`±WIN_VALUE`).
2. Threefold repetition (hash count ≥ 3) → draw.
3. No legal actions → draw.
4. `play_ply ≥ 300` → compare token counts for a terminal score.

For α–β, `WIN_VALUE = 10**9` with ply-adjusted preference for faster wins (`value -
ply` for us, `value + ply` against). For MCTS, `±1` terminal payoffs with `0.5` for
draw.

### 3.4 Evaluation function — `eval.py`

A single linear combination used by every iteration. Weights are tuned in §11 but a
sensible starting point is given below. Always computed from RED's perspective; the
negamax / MCTS wrappers sign-flip.

| #  | Feature                                                   | Initial weight |
|----|-----------------------------------------------------------|----------------|
| 1  | Token diff (`red_tokens − blue_tokens`)                   | +100           |
| 2  | Stack-count diff                                          | +20            |
| 3  | Weighted height (`Σ min(h, 4)` diff)                      | +15            |
| 4  | Edge-danger (our mass within 1–2 of an edge, per-dir)     | −25            |
| 5  | Threatened pieces diff (adjacent enemy with `h' ≥ h`)     | −40            |
| 6  | Attack potential diff (legal EATs available)              | +30            |
| 7  | Mobility diff (approx count of legal actions)             | +2             |
| 8  | Cascade reach diff (enemy mass reachable by our cascades) | +5             |
| 9  | Centre-ish control (Manhattan from (3.5, 3.5))            | +1             |
| 10 | Tempo (side-to-move term)                                 | +1             |

Features are cheap: piece-list iteration with short ray walks. No apply/undo in eval.

### 3.5 Placement-phase policy — `placement.py`

Only 8 moves total, but they frame the midgame. A 2-ply α–β over placement candidates
scored by the main eval plus a placement bonus (interior preference, spread penalty
for adjacency to own stacks, threat bonus for attacking enemy stacks). A tiny
hard-coded fallback book — e.g. Red opens `(3, 3)`, `(4, 4)`, `(3, 6)`, `(4, 1)` — is
used if search misbehaves or time runs short.

### 3.6 Time budget — `time_budget.py`

Single entry point consulted by every search loop. Per-move budget is
`time_remaining / max(30, expected_moves_left)` with a soft cap of ~5 s, hard cap
~8 s. Reserves ~20 s of the 180 s game budget for the endgame. All search iterations
use the same budget logic, so search strength can be compared like-for-like.

---

## 4. Iteration roadmap

| # | Label                          | Backend | Submit to Gradescope | Success criterion                                          |
|---|--------------------------------|---------|----------------------|------------------------------------------------------------|
| 1 | Fixed-depth α–β                | `ab`    | Yes                  | Beats template ≥ 90%, clears random tier (5 marks)         |
| 2 | ID α–β + TT + MVV-LVA          | `ab`    | Yes                  | Clears greedy tier (3 marks)                               |
| 3 | PVS + quiescence + killers     | `ab`    | Yes                  | Clears shallow-adversarial tier (3 marks), ≥ depth 5–6    |
| 4 | Vanilla UCT MCTS               | `mcts`  | No (regression risk) | ≥ 50% vs I3 at 5 s/move                                    |
| 5 | Heavy rollouts + RAVE + cutoff | `mcts`  | Yes                  | ≥ 60% vs I3                                                |
| 6 | MCTS-Solver + PUCT + reuse     | `mcts`  | Yes (final agent)    | ≥ 65% vs I3 at 5 s/move; stable over 200-game tournament   |

Each iteration reuses §3. Only the search backend changes. The old backend is kept in
source so head-to-head comparisons run on identical state/eval code — this is what
the report's performance plots will come from (§12).

---

## 5. Iteration 1 — Fixed-depth α–β (first shipped agent)

**Goal**: a simple, correct α–β that clears the random tier. No iterative deepening,
no transposition table, no move ordering.

### 5.1 What's new

- `search.py::negamax_fixed(board, depth, alpha, beta) -> (value, best_move)`.
- `program.py` calls `negamax_fixed` at a hard-coded depth of 3 for the play phase
  and delegates to `placement.py` during placement.

### 5.2 Algorithm

Textbook negamax α–β. Signed eval and signed terminal values make the sign flip a
single negation.

```python
def negamax_fixed(board, depth, alpha, beta):
    t = board.terminal()
    if t is not None:
        return signed_terminal_value(t, board.turn_color), None
    if depth == 0:
        return signed_eval(board), None

    best_move, best_val = None, -INF
    for move in board.legal_actions():
        board.apply(move)
        val, _ = negamax_fixed(board, depth - 1, -beta, -alpha)
        val = -val
        board.undo()
        if val > best_val:
            best_val, best_move = val, move
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break
    return best_val, best_move
```

### 5.3 What's deliberately absent

- No iterative deepening — depth is fixed at 3.
- No ordering — the generator's natural order is used.
- No TT, no quiescence, no killers.
- No per-move time polling (depth 3 is well inside the budget).

### 5.4 Why ship this

Depth-3 α–β with the full §3.4 eval already clears the random tier and most greedy
opponents. Shipping early flushes out integration bugs (agent protocol, placement,
time accounting) and gives every later iteration a hard regression baseline. A bug
that breaks I2 but was present in I1 will still pass the I1 test harness — so
iterations are compared against I1 as a floor.

### 5.5 Exit test

- `python -m referee agent agent` completes two full games with no exceptions.
- Local 100-game match against the random-legal opponent described in §11 wins ≥ 95%.
- Gradescope submission clears the random tier.

---

## 6. Iteration 2 — Iterative deepening + TT + basic ordering

**Goal**: depth becomes adaptive; repeated positions stop being re-searched; EAT
moves are tried first. This is where α–β starts to search meaningfully deeper.

### 6.1 What's new

- `search.py::iterative_deepening(board, time_budget)` wraps `negamax_ab`.
- `tt.py::TranspositionTable`: a bounded `dict[uint64, TTEntry]` with
  `(depth, value, flag ∈ {EXACT, LOWER, UPPER}, best_move, age)`. Cap ~2M entries;
  evict shallowest/oldest on overflow.
- `ordering.py::order_moves(board, tt_move)`:
  1. TT move first.
  2. EATs sorted by MVV-LVA: `score = 100 × target_h − attacker_h`.
  3. Cascades that push enemy mass off the board (cheap ray scan).
  4. Everything else in generator order.
- Zobrist-based position history (already in §3.1) is consulted at every node; if
  current hash count ≥ 2, treat as a forced draw (value 0) to stop the search chasing
  repetitions.

### 6.2 Algorithm

```python
def iterative_deepening(board, budget):
    best_move, best_val = None, 0
    depth = 1
    while budget.remaining() > budget.per_move_slice():
        val, move = negamax_ab(board, depth, -INF, +INF, ply=0)
        if budget.expired():
            break
        best_val, best_move = val, move
        depth += 1
    return best_move

def negamax_ab(board, depth, alpha, beta, ply):
    if budget.expired():
        raise SearchTimeout
    t = board.terminal()
    if t is not None:
        return ply_adjusted(t, ply)
    tt_entry = TT.get(board.hash)
    tt_move = tt_entry.best_move if tt_entry else None
    if tt_entry and tt_entry.depth >= depth:
        v, f = tt_entry.value, tt_entry.flag
        if f == EXACT: return v, tt_entry.best_move
        if f == LOWER and v > alpha: alpha = v
        if f == UPPER and v < beta:  beta  = v
        if alpha >= beta: return v, tt_entry.best_move
    if depth == 0:
        return signed_eval(board), None

    best_val, best_move = -INF, None
    alpha0 = alpha
    for move in order_moves(board, tt_move):
        board.apply(move)
        val, _ = negamax_ab(board, depth - 1, -beta, -alpha, ply + 1)
        val = -val
        board.undo()
        if val > best_val:
            best_val, best_move = val, move
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break
    flag = EXACT if alpha0 < best_val < beta else (LOWER if best_val >= beta else UPPER)
    TT.put(board.hash, TTEntry(depth, best_val, flag, best_move, age=play_ply))
    return best_val, best_move
```

### 6.3 Time management

`budget.expired()` raises `SearchTimeout` inside `negamax_ab`. The outer
`iterative_deepening` catches it, discards the partial depth, and returns the best
move from the last *completed* depth. This is why ID is valuable: a depth-6 search
that times out still has a depth-5 move in hand.

### 6.4 Exit test

- Clears greedy tier on Gradescope.
- Local 100-game match wins ≥ 80% vs I1.
- Reaches depth 4–5 consistently on a mid-game position within 3 s.

---

## 7. Iteration 3 — PVS + quiescence + killers + history + rich eval

**Goal**: squeeze the last meaningful gains out of α–β. This is the "benchmark
agent" the spec recommends having before attempting MCTS.

### 7.1 What's new

- **Principal Variation Search (PVS)** replaces plain α–β: the first child is
  searched with the full window `(−β, −α)`, subsequent children with a null window
  `(−α−1, −α)`. On a null-window fail-high, the child is re-searched with the full
  window.
- **Quiescence search** at the horizon. Only "noisy" moves are explored:
  - Every `EatAction`.
  - Every `CascadeAction` whose ray touches at least one enemy token or would push an
    enemy stack off the edge.
  - Stand-pat using `signed_eval(board)`; cut immediately if stand-pat ≥ β.
  - Delta pruning: skip captures that can't raise alpha even in the best case.
  Capped at depth 6 plies from the horizon to prevent pathological explosion.
- **Killer-move heuristic**: per ply, two non-capture moves that caused a β-cutoff.
  Tried after EATs, before the rest.
- **History heuristic**: `history[piece, from, to, action_kind] += depth²` on every
  β-cutoff; used as the tiebreaker in `ordering.py` for quiet moves.
- **Aspiration windows** (optional, easy): at iteration `d ≥ 3`, start with
  `alpha = prev_val − 50, beta = prev_val + 50` and widen on fail-low/high. Saves
  ~15–25% of nodes when the eval is stable between depths.
- **Enhanced eval**: promote features 4–8 from §3.4 with proper tuning (§11). Add:
  - Repetition avoidance bonus: if our side is ahead on tokens, penalise moves that
    push the hash count toward 3.
  - Placement-phase endgame awareness: if `play_ply > 280`, boost the token diff
    weight so the search prefers drawn-toward-win over speculative structure.

### 7.2 Why it matters

Empirically PVS + TT reaches 1.5–2 plies deeper than plain α–β in the same time
budget when move ordering is good. Quiescence plugs the most embarrassing horizon
effect in Cascade: the search stopping one ply before a deciding CASCADE.

### 7.3 Risks and mitigations

- **Quiescence explosion**: Cascade's branching in "noisy" positions can stay high
  (every stack with `h ≥ 2` has 4 cascades). Mitigation: cap quiescence depth and
  require a cascade to touch enemy mass to be admitted.
- **TT mis-ordering under PVS**: a null-window hit must not be stored as `EXACT`.
  Make flag bookkeeping explicit and unit-test against hand-built positions.

### 7.4 Exit test

- Clears shallow-adversarial tier on Gradescope.
- Local 100-game match wins ≥ 65% vs I2.
- Depth 5–6 reached within the 5 s budget on representative mid-game positions.
- `cProfile` shows > 60% of time in eval + move gen, < 20% in TT/ordering overhead.

**Decision point.** If I3 plays visibly well and there's > 6 days left, proceed to I4.
If not, freeze I3 as the submitted agent and write the report around the α–β
progression. MCTS is an "upside play"; the benchmark agent is the safety net.

---

## 8. Iteration 4 — Vanilla UCT MCTS

**Goal**: working UCB1 MCTS with random rollouts. Correctness first, strength later.
Not submitted to Gradescope (regression risk — random rollouts are weak on Cascade).

### 8.1 What's new

All in `mcts.py`.

- **Node**:

  ```python
  @dataclass
  class Node:
      parent: "Node | None"
      incoming_move: Action | None
      children: dict[Action, "Node"]     # populated lazily on expansion
      untried: list[Action] | None       # None until first visit
      visits: int
      total_value: float                 # sum of terminal outcomes from this node's
                                         # perspective, i.e. from the player *about to
                                         # move at `self`*
      terminal_value: float | None       # ±1 or 0 if proven terminal (I6), else None
  ```

- **Main loop**:

  ```python
  def search(root_board, budget):
      root = Node(parent=None, incoming_move=None, children={}, untried=None,
                  visits=0, total_value=0.0)
      while budget.remaining() > 0:
          board = root_board.clone_for_mcts()
          node = select(root, board)
          node = expand(node, board)
          reward = rollout(board)                  # ±1 / 0
          backprop(node, reward)
      return best_child(root).incoming_move
  ```

- **Selection (UCB1)**:

  ```python
  def uct_score(child, parent_visits, c=math.sqrt(2)):
      if child.visits == 0:
          return math.inf
      return (child.total_value / child.visits
              + c * math.sqrt(math.log(parent_visits) / child.visits))
  ```

  Picked child's `incoming_move` is applied to `board`.

- **Expansion**: on first visit, populate `node.untried` by calling the legal-action
  generator. Pop one action, apply to `board`, create the child, return it.
- **Rollout**: play random legal moves until terminal, a repetition-draw is forced,
  or a rollout depth cap (say 80 play plies) is hit. Cap is essential — Cascade has
  no mandatory captures, so random play can stall indefinitely.
- **Backpropagation**: walk up the parent chain, flipping reward sign at each step
  (two-player zero-sum).
- **Final move choice**: **robust child** (max `visits`) at root, not max value.
  Ties broken by `total_value / visits`.

### 8.2 Cloning vs undo

The naive "clone the board on every iteration" is simple but slow. Because I1–I3
already have a fast in-place `apply/undo`, MCTS can keep the same pattern: record the
path of applied moves on descent, `undo` them on ascent after the rollout. A rollout
needs its own undo trail too, or a single `clone()` right before rollout starts
(good trade: one clone per iteration instead of ~200 undo records). Default:
one clone at rollout start.

### 8.3 Exit test

- No exceptions over 500 simulated games.
- Local 50-game match vs I3 at 5 s/move: MCTS reaches ≥ 30% win rate. (Random
  rollouts are weak; this is a correctness-of-plumbing check, not a strength target.)

If this iteration is visibly worse than I3, proceed to I5 — the remaining iterations
close the gap.

---

## 9. Iteration 5 — Heavy rollouts + RAVE + early termination

**Goal**: replace random rollouts with heuristic ones; share information across
siblings via RAVE; cut rollouts short with the eval. This is where MCTS starts to
play like a real agent.

### 9.1 What's new

- **Heavy playout policy** (`policy.py::rollout_policy`):
  - If any legal move immediately wins (eliminates the opponent), take it.
  - Else if any opposing move on our next turn would immediately win and we have a
    move that prevents it, take the best such defensive move.
  - Else sample an action with probability proportional to an ε-soft heuristic
    score: `score(move) = MVV-LVA for EAT + pushed-mass for CASCADE + centre bonus
    for MOVE`. Softmax with temperature τ = 1.0.
  This is cheap — one pass through the legal list with the same scoring used by
  `ordering.py` — and produces playouts that actually resemble play.
- **Early rollout termination**: cap rollout depth at D = 25 play plies. If the cap
  is hit, return `tanh(signed_eval(board) / scale)` as the "reward" in `[-1, +1]`
  instead of a random guess. `scale ≈ 500` is tuned so a 5-token lead yields ~0.76.
- **RAVE / AMAF**: per node, maintain a second pair `(rave_visits, rave_value)` per
  action. On backprop, update not just the action taken but also every action that
  occurred anywhere later in the simulation from the same side to move ("all moves
  as first"). Combine in selection:

  ```python
  beta = rave_visits / (rave_visits + visits + 4 * rave_visits * visits * b2)
  q    = (1 - beta) * (value / visits) + beta * (rave_value / rave_visits)
  score = q + c * sqrt(log(parent_visits) / visits)
  ```

  `b2 ≈ 1e-5` controls how quickly RAVE is phased out as real visits accumulate.
  RAVE is especially effective when actions are reorderable, which is often true
  across Cascade turns (the same EAT/MOVE can arise in many lines).
- **Virtual loss removed/skipped** — single-threaded by spec, so no need.
- **First-play urgency (FPU)**: un-visited children get a score of
  `parent_value − 0.25` instead of infinity. This discourages blindly expanding
  every sibling before exploiting promising ones.

### 9.2 Exit test

- Local 100-game match vs I3: MCTS at 5 s/move wins ≥ 60%.
- Rollouts/second on a mid-game position: ≥ 3000 (with the heavy policy, this
  requires the fast internal board — if it isn't, profile and fix).
- Gradescope submission: maintains or improves on I3's tier.

---

## 10. Iteration 6 — Sophisticated MCTS (final submitted agent)

**Goal**: the final agent. Combines MCTS-Solver, PUCT priors from a hand-crafted
policy, tree reuse between moves, and an α–β hybrid at promising leaves.

### 10.1 What's new

#### (a) MCTS-Solver — prove terminal values

A node can be *proven* (`terminal_value ∈ {−1, 0, +1}`) when its entire subtree is
resolved. Once proven:

- A child proven as a loss for the mover (value `−1` from the mover's perspective)
  means the parent has at least one winning move — propagate `+1`.
- A parent whose *every* child is proven a win for the mover (each child's value
  `+1` from mover-of-child's perspective, i.e. `−1` from the parent's) is itself
  proven `−1`.
- Selection skips proven-loss children (they're known-refuted) and proven-draws are
  treated normally until a win is found.

Reference: Winands, Björnsson, Saito (2008) "Monte-Carlo Tree Search Solver". This
alone visibly sharpens endgame play, which in Cascade starts around token counts
≤ 8–10.

#### (b) PUCT with hand-crafted priors

Selection switches from UCB1 to AlphaZero-style PUCT:

```python
score(child) = Q(child) + c_puct * P(child) * sqrt(parent_visits) / (1 + child.visits)
```

`P(child)` is a prior over legal actions. With no NN available (stdlib + NumPy only),
priors come from `policy.py::prior(board)` — the same heuristic scorer used in
rollouts, normalised via softmax. `c_puct` ≈ 1.5–2.5 (tuned). PUCT with a decent
prior is strictly better than UCB1 when the branching factor is in the 30–80 range
typical of Cascade midgame: it focuses simulations on plausible moves from visit 1
instead of after many exploratory pulls.

#### (c) Tree reuse between turns

After the agent plays move `m_us` and the opponent plays `m_them`, descend to the
grandchild node `root.children[m_us].children[m_them]`, detach it, and use it as the
root of the next search. All visit counts, Q-values, and RAVE statistics from the
previous turn are retained. Orphaned subtrees are released.

Edge cases:
- The opponent's move might not be in the tree (we never expanded it). In that case
  fall back to a fresh root, but still seed the TT with the prior.
- The hash of the new root should match the current `board.hash`; assert this —
  mismatch implies an update bug.

Tree reuse can cut per-move thinking time by 30–60% in the midgame.

#### (d) α–β hybrid at expansion

When a leaf is expanded and its parent has > `N_HYBRID = 32` visits, run a depth-2
α–β (`search.negamax_ab` from I3 at depth 2) from that leaf and use the result as
the initial `(visits=1, total_value=value/scale)` of the new child instead of a
rollout. The search result is converted to `[-1, +1]` via the same `tanh` used in
early termination.

This is deliberately shallow (depth 2) — it's a "smart rollout", not a full search
— but it injects real tactical information where random+heuristic rollouts are
thinnest, and it exploits the α–β machinery already built in I1–I3.

#### (e) Decisive / anti-decisive move detection

At every node, before generic selection:
- If any move wins immediately, play it (don't even expand; mark the node proven).
- Else if the opponent has exactly one response that wins against us in one ply,
  prefer the move that removes it.

These two checks are pure legal-move iteration, no search. They account for a
surprisingly large fraction of late-game errors in vanilla MCTS.

#### (f) Time management tuned for MCTS

- Don't divide budget evenly. Spend more on positions where the root's top-two
  children are close in visit count; cut early if the top child has ≥ 3× the
  visits of the next.
- Dirichlet noise at the root (`α = 0.3, ε = 0.1`) **disabled** — we're not
  self-playing, and noise costs real-game strength.

#### (g) Memory hygiene

Tree reuse means nodes persist. Cap total node count at ~200k; on overflow, drop
the subtree rooted at the least-visited root child (MCTS-style "tree trimming").
Each `Node` is ~200 bytes — 200k nodes is ~40 MB, comfortably inside the 250 MB
budget after NumPy arrays and the TT.

### 10.2 Integration with the α–β codepath

`program.py` keeps the I3 α–β backend compiled in. In the final submission it is:
- The depth-2 "smart rollout" inside MCTS expansion.
- A fallback used when `budget.remaining() < 0.5 s` at move time, since α–β with the
  TT still warm from the previous turn returns a sane move in tens of milliseconds.

Both backends share §3's state, eval, ordering, zobrist, and TT, so the composition
is essentially free.

### 10.3 Exit test

- ≥ 65% win rate vs I3 over a 200-game tournament, at 5 s/move each, on the local
  harness.
- ≥ 70% vs the I1 baseline (sanity regression test).
- Maintains Gradescope tier from I5 or better.
- No timeouts over 500 games.
- Node count and memory stay bounded under the caps.

---

## 11. Testing and evaluation harness (shared across iterations)

- **Local round-robin runner** (outside `agent/`, not submitted): subprocess-calls
  `python -m referee <redpkg> <bluepkg>`, collects result, aggregates into an
  Elo-like table. Used nightly during M2–M6.
- **Fixed baselines** kept in sibling packages:
  - `agent_random/` — uniform random over legal moves.
  - `agent_greedy/` — 1-ply α–β with just the token-diff feature.
  - `agent_ab_frozen/` — a snapshot of I3 (depth-capped, 5 s/move) — the benchmark.
  Each MCTS iteration is measured against *all three*, not just the previous
  iteration — prevents "tunnel vision" where a weight set beats the recent self
  but regresses against simpler opponents.
- **Unit tests** in `tests/`:
  - Board legality: hand-built positions covering edge-of-board cascades, push
    chains, eat-equal-height, placement-adjacency rule, self-elimination cascade.
  - Apply/undo round-trip: for 1000 random legal move sequences, hash before/after
    must match.
  - Zobrist incremental vs full recomputation on a random 100-move trajectory.
  - Perft-style node counts on depth-3 from a fixed mid-game position; cross-check
    against a reference enumeration using `referee/game/board.py`.
- **Eval-weight tuning** (offline, not in the submitted binary):
  - Coarse sweep: 3^10 grid is infeasible, so tune in two passes — first features
    1–3 (material), then 4–8 (positional) with 1–3 frozen.
  - Self-play tournament per candidate weight set; winner replaces the champion
    using Elo with a margin test (min 40 games, score ≥ +50 Elo).
  - Optional TDLeaf(λ) polish if there's week-of-deadline time.
- **Profiling**: `cProfile` on 10 s of search from 3 representative positions (early
  mid, mid, endgame). Target: eval + move-gen ≥ 60% of CPU time. Below that, fix
  the hot path before starting the next iteration.

---

## 12. Report outline (drafted alongside the code)

Target 6 pages. Each bracketed figure is a *concrete* artefact — they'll exist
because the harness already produces them.

1. **Approach overview** (½ page). One paragraph on α–β benchmark → MCTS final.
2. **State representation & move generation** (½). Why a custom `int8` board, signed
   encoding, piece lists, Zobrist, in-place apply/undo.
3. **α–β benchmark agent** (1½). PVS, TT, quiescence, killers/history. [Figure:
   nodes/sec and max depth vs iteration.]
4. **Evaluation function** (1). Feature list with strategic motivation. Weight-tuning
   protocol. [Figure: per-feature ablation win-rate.]
5. **MCTS** (2). UCT → heavy rollouts + RAVE + early termination → MCTS-Solver +
   PUCT + tree reuse + α–β hybrid. [Figure: win-rate vs I3 per MCTS iteration.]
   Credit the relevant papers: Coulom 2006 (MCTS), Kocsis & Szepesvári 2006 (UCT),
   Gelly & Silver 2007 (RAVE), Winands et al. 2008 (MCTS-Solver), Silver et al.
   2017 (PUCT).
6. **Placement policy** (¼). 2-ply α–β + placement bonus; small opening book.
7. **Performance evaluation** (¼). Round-robin results, depth-vs-time curve,
   Gradescope tier progression.

---

## 13. Risks and mitigations

- **MCTS regresses against I3**: mitigation — the α–β backend is always compiled
  in and can be selected by `SEARCH_BACKEND="ab"` in the final submission. Decision
  criterion at each MCTS iteration: ship *only if* it beats I3 over ≥ 200 games.
- **Time-budget blowouts**: single `time_budget.py` polled by every backend, every
  loop. Hard kill-switch returns the best move from the last completed MCTS step
  (or iterative-deepening layer).
- **Memory**: TT capped at ~2M entries; MCTS tree capped at ~200k nodes. Together
  that's ~120 MB worst case, leaving headroom for NumPy and the rest.
- **Referee divergence**: the apply/undo logic must exactly mirror
  `referee/game/board.py`. Perft + random-trajectory tests in §11 catch this.
- **Eval over-fitting to self-play**: weights are validated against
  `agent_random` / `agent_greedy` / `agent_ab_frozen` — a weight set that beats
  ourselves 55–45 but loses to greedy is rejected.
- **Python-level slowness**: if node rates fall below ~30k/s for α–β or ~2k
  rollouts/s for MCTS, profile before adding features. NumPy vectorisation of the
  eval (mask-based features 4, 8) is the usual fix.
