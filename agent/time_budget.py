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