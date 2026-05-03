# variants/

Dev-only benchmark agents. Each subdirectory is a referee-compatible module
that wires the shared `agent.core` state to a *specific* search backend (or
implements a baseline policy directly), so the bench harness can run multiple
strategies side-by-side as distinct players.

**Not submitted.** Per assignment spec, only `agent/` + `team.py`
+ `report.pdf` ship to Gradescope. `variants/` is purely local scaffolding.

## Why this directory exists

`python -m referee X Y` needs `X` and `Y` to be **two separately importable
modules**, each with an `Agent` class. With only `agent/` you can play it
against itself, but you can't say "ID α–β vs PVS" or "MCTS vs greedy" in the
same match — those need different module paths. `variants/` is where those
extra entry points live.

## Current variants

| path                       | what it runs                                         |
|----------------------------|------------------------------------------------------|
| `variants/ab1/`            | I1: depth-3 fixed negamax (`agent.search.ab_fixed`)  |
| `variants/ab2/`            | I2: ID α–β + TT + ordering (`agent.search.ab_id`)    |
| `variants/ab3/`            | I3: PVS + quiescence + killers + history + aspiration windows (`agent.search.pvs`) |

Planned, not yet built: `variants/random_bot/`, `variants/greedy/`,
`variants/mcts_uct/`, `variants/ab_frozen_v3/`. See
`docs/implementation_plan.md` §11.

## Anatomy of a thin shim

Each existing variant is ~30 lines. Structure:

```
variants/<name>/
├── __init__.py     # one line: `from .program import Agent`
└── program.py      # Agent class wired to one specific backend
```

`program.py` imports `agent.core.*` for the shared state/eval/placement and
one specific backend from `agent.search.*` (or implements its own policy).
The `Agent` class follows the interface in `docs/assignment_spec.md` §2.1:
`__init__(color, **referee)`, `action(**referee) -> Action`,
`update(color, action, **referee)`.

Take `variants/ab2/program.py` as the template if you're adding a new
backend-shim variant.

## Running them

Same syntax as any referee module:

```bash
# Single game, ab3 (Red) vs ab1 (Blue), 30 s/agent
python -m referee -v 0 -t 30 variants.ab3 variants.ab1

# 100-game bench with balanced colours, 4-way parallel
python tools/bench.py --red variants.ab3 --blue variants.ab1 -n 100 -j 4
```

Slash form also works: `python -m referee variants/ab3 variants/ab1`.

## Adding a new variant

Backend-shim variant (the common case — you wrote a new search in
`agent/search/<name>.py` and want to bench it):

1. Copy `variants/ab2/` to `variants/<name>/`.
2. Edit `program.py` — change the `agent.search.*` import and the call site.
3. Smoke test: `python -m referee variants.<name> variants.ab2`.

Standalone-policy variant (random, greedy, etc. — doesn't need a search
backend):

1. Make `variants/<name>/__init__.py` and `program.py`.
2. Implement `Agent` directly in `program.py`. You can still import
   `agent.core.board.GameState` for legal-move generation.

## Shared core vs frozen snapshots — read this before benching

Every variant under `variants/` listed above is a **thin shim**. It does
`from agent.core.board import GameState`, `from agent.core.eval import
evaluate`, etc. — all running against the *live* `agent/core/`.

Implication: when you tune `agent/core/eval.py` weights for I4 (MCTS),
**every shim shifts with it**. The `ab1` you benched two weeks ago doesn't
play the same game today. Your old win-rate numbers don't compare anymore.

For most day-to-day work this is fine — you want the comparison to reflect
the latest core. But when you publish a number the report will cite, snapshot
the relevant variant by **copying the entire `agent/` tree**:

```bash
cp -r agent variants/ab_frozen_v3
```

The result is a self-contained directory with its own private
`variants/ab_frozen_v3/core/` and `variants/ab_frozen_v3/search/`. Internal
imports use relative paths (`from .core.eval import ...`), so nothing breaks
on copy. Now `variants/ab_frozen_v3` is immune to drift — it plays exactly
the way I3 played at freeze time, forever.

Do this **once per shipped iteration** (after I3, after I5, etc.) so future
MCTS work has a stable yardstick.

## Naming convention

- `ab1`, `ab2`, `ab3`, `mcts_uct`, `mcts_heavy`, `mcts_solver` — match the
  module name in `agent/search/` they wrap.
- `random_bot`, `greedy` — descriptive name for standalone baselines.
- `ab_frozen_v3` — `<thing>_frozen_v<iteration>` for full snapshots. Bumping
  the version number on each freeze keeps old snapshots intact.
