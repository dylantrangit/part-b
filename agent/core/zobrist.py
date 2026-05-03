import numpy as np

MAX_HEIGHT_BUCKET = 12

_rng = np.random.default_rng(2026)

# Z[flat_idx, colour, height_bucket]
# colour: 0 = red, 1 = blue
Z = _rng.integers(
    low=0,
    high=np.iinfo(np.uint64).max,
    size=(64, 2, MAX_HEIGHT_BUCKET + 1),
    dtype=np.uint64,
)

SIDE_TO_MOVE_KEY = np.uint64(
    _rng.integers(
        low=0,
        high=np.iinfo(np.uint64).max,
        dtype=np.uint64,
    )
)