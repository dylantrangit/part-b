# agent/core/

The shared substrate every search backend builds on: state representation,
legal-move generation, evaluation, move ordering, transposition table,
placement policy, time budget, and Zobrist hashing.

## Modules

| file              | what it owns                                                              |
|-------------------|---------------------------------------------------------------------------|
| `board.py`        | `GameState` — int8 grid, piece lists, in-place apply/undo, legal-action generator, terminal detection. The hot path. |
| `zobrist.py`      | 64-bit hash table seeded once at import. XOR-incremental, used for the TT and the threefold-repetition counter. |
| `eval.py`         | `evaluate(state)` — linear combination of ten features, returned from Red's perspective. Search code negates for Blue. |
| `ordering.py`     | `order_moves` (I2) and `order_moves_pvs` (I3); `generate_noisy_actions` for quiescence; helpers for MVV-LVA and noisy-cascade detection. |
| `tt.py`           | `TranspositionTable` — bounded `dict[uint64, TTEntry]` with depth-preferred replacement. |
| `placement.py`    | `choose_placement_action(state)` — opening / placement-phase policy used by every backend. |
| `time_budget.py`  | `TimeBudget`, `per_move_budget`, `SearchTimeout`. The single source of truth for time accounting. |

## Design rule

**Backend-agnostic only.** Anything in `core/` must be usable by α–β,
quiescence, MCTS, and any future backend without modification. If a piece
of code only makes sense for one search algorithm — node structs, UCB
selection, RAVE statistics, killer-move tables, PVS-specific re-search
logic — it lives in `agent/search/<backend>.py`, not here.

The two exceptions worth flagging because they're easy to misread:

- **`order_moves` and `order_moves_pvs` in `ordering.py`** look
  search-specific (different backends prefer different ordering), but the
  *primitives* — MVV-LVA scoring, noisy-cascade detection, history-table
  sorting — are shared. The orderers compose those primitives in different
  ways and live here so a new backend can mix-and-match without copy-paste.
- **`tt.py`** is a generic key-value store. Backends decide what flag
  semantics (EXACT/LOWER/UPPER) mean for them. MCTS doesn't have to use it
  at all.

## The drift hazard

`core/` is shared by `agent/program.py` *and* every shim under `variants/`.
Changing `eval.py` weights or `board.py` move ordering shifts the playing
strength of every variant simultaneously. Two consequences:

1. Old win-rate numbers stop being comparable. If you publish "I3 beats I2
   65% of the time" and then tune `eval.py`, both sides moved — you'd need
   to re-measure.
2. Frozen baselines (e.g. `variants/ab_frozen_v3/`) protect against this by
   carrying a *copy* of `core/`. See `variants/README.md` for the snapshot
   procedure.

Day-to-day: tune freely, re-measure when it matters. Snapshot when a number
needs to outlive future tuning.

## State invariants worth preserving

These are easy to break and expensive to debug. Whenever `board.py`
changes, check that:

- After every `apply` followed by `undo`, the grid, piece lists, scalar
  counts, zobrist hash, play history, and turn colour all match their
  pre-apply values exactly. The undo stack should be empty if you started
  empty.
- The zobrist hash computed incrementally matches the hash recomputed from
  scratch over the grid + side-to-move. Drift here corrupts the TT silently.
- `legal_actions()` and `has_any_legal_action()` agree — the second must be
  exactly "the first yields at least once."
- Placement-phase positions are *not* recorded in `play_history` (per spec:
  threefold repetition only counts play-phase positions).

`tests/` is where these invariants belong as automated checks (not yet built).
