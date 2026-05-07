# agent/

The shipping package. **This directory plus `team.py` and `report.pdf` is the
entire Gradescope submission**. It must be
self-contained — no imports from outside `agent/` except the provided
`referee` module.

## Layout

```
agent/
├── __init__.py     # re-exports Agent (referee imports this)
├── program.py      # the Agent class — interface to the referee
├── core/           # shared substrate (board, eval, placement, ...)
│   └── README.md
└── search/         # interchangeable search backends
    ├── ab_fixed.py # I1
    ├── ab_id.py    # I2
    └── pvs.py      # I3 (current)
```

`program.py` is a thin shim — it owns the referee-facing `Agent` class and
delegates the actual move selection to one backend from `agent/search/`.
Switching iterations is a one-line import change.

## The Agent interface

`agent/program.py::Agent` implements the three methods the referee calls:

- `__init__(self, color, **referee)` — once per game; create the internal
  `GameState` and any per-game caches (e.g. transposition table).
- `action(self, **referee) -> Action` — pick a move. Reads `time_remaining`
  from `**referee`, derives a per-move budget, and calls the active search
  backend.
- `update(self, color, action, **referee)` — apply the just-played action to
  the internal state. Called for *both* players' moves.

The referee creates one `Agent` instance per player in its own subprocess.

## Active backend

The backend currently wired into `program.py` is `agent.search.pvs`. To swap
it for a different one, change two lines:

```python
# in agent/program.py
from .search.pvs import iterative_deepening_pvs as iterative_deepening
# ...
move = iterative_deepening(self.state, budget, self.tt)
```

There is no runtime config for this. Compile-time selection is intentional —
the submitted agent should be one specific configuration, not a switchboard
of backends loaded by environment variables (Gradescope sets none).

## Running locally

```bash
# Self-play
python -m referee agent agent

# Against a frozen baseline
python -m referee agent variants.ab1

# 100-game bench against the previous iteration
python tools/bench.py --red agent --blue variants.ab2 -n 100 -j 4
```

## Constraints (from the spec)

- 180 s CPU per agent per game.
- 250 MB peak memory per agent per game.
- Python 3.12, stdlib + NumPy only. No other libraries.
- No threads, no disk I/O, no network.

The time-management code in `agent/core/time_budget.py` honours the CPU limit
by polling `budget.expired()` inside every search loop and aborting cleanly.

## What NOT to add to `agent/`

- **Tests** — they live in `tests/`, not in the shipping package.
- **Bench scripts, snapshots, baseline opponents** — those go in `tools/` and
  `variants/`.
- **Anything that imports from outside `agent/` and `referee/`** — it will
  fail on Gradescope.
