from dataclasses import dataclass
from typing import Optional


EXACT = 0
LOWER = 1
UPPER = 2


@dataclass
class TTEntry:
    depth: int
    value: float
    flag: int
    best_move: Optional[object]
    age: int


class TranspositionTable:
    def __init__(self, max_entries=2_000_000):
        self._table = {}
        self._max = max_entries

    def get(self, key):
        return self._table.get(key)

    def put(self, key, entry):
        existing = self._table.get(key)
        if existing is not None:
            # Depth-preferred replacement: keep the deeper entry, but accept a
            # shallower one if it's younger (newer search context).
            if existing.depth > entry.depth and existing.age >= entry.age:
                return
        if len(self._table) >= self._max:
            self._evict()
        self._table[key] = entry

    def _evict(self):
        # Drop ~10% of entries with the smallest (depth, age) — shallowest and
        # oldest go first.
        target_size = self._max - self._max // 10
        items = sorted(
            self._table.items(),
            key=lambda kv: (kv[1].depth, kv[1].age),
        )
        drop = len(self._table) - target_size
        for k, _ in items[:drop]:
            self._table.pop(k, None)

    def clear(self):
        self._table.clear()

    def __len__(self):
        return len(self._table)
