import time


class SearchTimeout(Exception):
    pass


def per_move_budget(time_remaining, expected_moves_left):
    budget = time_remaining / max(30, expected_moves_left)

    # reserve ~20 s of total budget for endgame
    usable_budget = max(0.0, time_remaining - 20.0)
    budget = min(budget, usable_budget)

    # soft cap ~5 s
    budget = min(budget, 5.0)

    # hard cap ~8 s
    budget = min(budget, 8.0)

    if budget < 0.01:
        budget = 0.01

    return budget


class TimeBudget:
    def __init__(self, seconds):
        self._total = seconds
        self._deadline = time.monotonic() + seconds

    def remaining(self):
        return max(0.0, self._deadline - time.monotonic())

    def expired(self):
        return time.monotonic() >= self._deadline

    def per_move_slice(self):
        # If less than this is left, don't bother starting another ID depth.
        return max(0.05, self._total * 0.1)