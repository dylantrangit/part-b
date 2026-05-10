from referee.game import PlayerColor


# Eval weights:
# - STACK_DIFF kept low: more stacks of the same total mass means each is
#   shorter and easier to EAT, so over-rewarding raw stack count would push
#   us toward "spread thin" positions.
# - WEIGHTED_HEIGHT high: captures stack concentration. A height-3 stack is
#   materially worth more than three height-1 tokens despite identical
#   token count.
# - THREATENED deep: vulnerable pieces dominate Cascade tactics, and the
#   magnitude needs to be visible at the rollout-cutoff eval used by MCTS.
TOKEN_DIFF_WEIGHT = 100
STACK_DIFF_WEIGHT = 10
WEIGHTED_HEIGHT_WEIGHT = 30
EDGE_DANGER_WEIGHT = -25
THREATENED_WEIGHT = -60
ATTACK_POTENTIAL_WEIGHT = 30
MOBILITY_WEIGHT = 2
CASCADE_REACH_WEIGHT = 5
CENTRE_WEIGHT = 1
TEMPO_WEIGHT = 1
REPETITION_AVOID_WEIGHT = 50
ENDGAME_EMPHASIS_WEIGHT = 50
ENDGAME_RAMP_PLY = 280
ENDGAME_RAMP_LEN = 20


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def evaluate(state):
    score = 0
    score += TOKEN_DIFF_WEIGHT * token_diff(state)
    score += STACK_DIFF_WEIGHT * stack_diff(state)
    score += WEIGHTED_HEIGHT_WEIGHT * weighted_height_diff(state)
    score += EDGE_DANGER_WEIGHT * edge_danger_diff(state)
    score += THREATENED_WEIGHT * threatened_diff(state)
    score += ATTACK_POTENTIAL_WEIGHT * attack_potential_diff(state)
    score += MOBILITY_WEIGHT * mobility_diff(state)
    score += CASCADE_REACH_WEIGHT * cascade_reach_diff(state)
    score += CENTRE_WEIGHT * centre_diff(state)
    score += TEMPO_WEIGHT * tempo(state)
    score += REPETITION_AVOID_WEIGHT * repetition_avoidance(state)
    score += ENDGAME_EMPHASIS_WEIGHT * endgame_emphasis(state)
    return score


def token_diff(state):
    return state.red_tokens - state.blue_tokens


def stack_diff(state):
    return state.red_stacks - state.blue_stacks


def weighted_height_diff(state):
    red_score = 0
    for flat_idx in state.red_pieces:
        coord = state.coord_from_flat(flat_idx)
        red_score += min(state.get_height(coord), 4)

    blue_score = 0
    for flat_idx in state.blue_pieces:
        coord = state.coord_from_flat(flat_idx)
        blue_score += min(state.get_height(coord), 4)

    return red_score - blue_score


def _edge_danger(grid, pieces):
    total = 0
    for flat_idx in pieces:
        r = flat_idx // 8
        c = flat_idx % 8
        h = abs(int(grid[r, c]))
        edges = 0
        if r <= 1:
            edges += 1
        if r >= 6:
            edges += 1
        if c <= 1:
            edges += 1
        if c >= 6:
            edges += 1
        total += h * edges
    return total


def edge_danger_diff(state):
    grid = state.grid
    return (
        _edge_danger(grid, state.red_pieces)
        - _edge_danger(grid, state.blue_pieces)
    )


def _threatened_count(grid, our_pieces, enemy_is_negative):
    count = 0
    for flat_idx in our_pieces:
        r = flat_idx // 8
        c = flat_idx % 8
        h = abs(int(grid[r, c]))
        for dr, dc in _DIRS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                v = int(grid[nr, nc])
                if v == 0:
                    continue
                is_enemy = (v < 0) if enemy_is_negative else (v > 0)
                if is_enemy and abs(v) >= h:
                    count += 1
                    break
    return count


def threatened_diff(state):
    grid = state.grid
    red_threatened = _threatened_count(grid, state.red_pieces, enemy_is_negative=True)
    blue_threatened = _threatened_count(grid, state.blue_pieces, enemy_is_negative=False)
    return red_threatened - blue_threatened


def _attack_count(grid, our_pieces, enemy_is_negative):
    count = 0
    for flat_idx in our_pieces:
        r = flat_idx // 8
        c = flat_idx % 8
        h = abs(int(grid[r, c]))
        for dr, dc in _DIRS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                v = int(grid[nr, nc])
                if v == 0:
                    continue
                is_enemy = (v < 0) if enemy_is_negative else (v > 0)
                if is_enemy and h >= abs(v):
                    count += 1
    return count


def attack_potential_diff(state):
    grid = state.grid
    red_attacks = _attack_count(grid, state.red_pieces, enemy_is_negative=True)
    blue_attacks = _attack_count(grid, state.blue_pieces, enemy_is_negative=False)
    return red_attacks - blue_attacks


def _mobility(grid, our_pieces, friendly_is_positive):
    count = 0
    for flat_idx in our_pieces:
        r = flat_idx // 8
        c = flat_idx % 8
        h = abs(int(grid[r, c]))
        for dr, dc in _DIRS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                v = int(grid[nr, nc])
                if v == 0:
                    count += 1
                else:
                    is_friendly = (v > 0) if friendly_is_positive else (v < 0)
                    if is_friendly:
                        count += 1
                    elif h >= abs(v):
                        count += 1
            if h >= 2:
                count += 1
    return count


def mobility_diff(state):
    grid = state.grid
    return (
        _mobility(grid, state.red_pieces, friendly_is_positive=True)
        - _mobility(grid, state.blue_pieces, friendly_is_positive=False)
    )


def _cascade_reach(grid, our_pieces, enemy_is_negative):
    total = 0
    for flat_idx in our_pieces:
        r = flat_idx // 8
        c = flat_idx % 8
        h = abs(int(grid[r, c]))
        if h < 2:
            continue
        for dr, dc in _DIRS:
            for step in range(1, h + 1):
                nr = r + dr * step
                nc = c + dc * step
                if not (0 <= nr < 8 and 0 <= nc < 8):
                    break
                v = int(grid[nr, nc])
                if v == 0:
                    continue
                is_enemy = (v < 0) if enemy_is_negative else (v > 0)
                if is_enemy:
                    total += abs(v)
    return total


def cascade_reach_diff(state):
    grid = state.grid
    return (
        _cascade_reach(grid, state.red_pieces, enemy_is_negative=True)
        - _cascade_reach(grid, state.blue_pieces, enemy_is_negative=False)
    )


def _centre_score(pieces):
    total = 0.0
    for flat_idx in pieces:
        r = flat_idx // 8
        c = flat_idx % 8
        total -= abs(r - 3.5) + abs(c - 3.5)
    return total


def centre_diff(state):
    return _centre_score(state.red_pieces) - _centre_score(state.blue_pieces)


def tempo(state):
    return 1 if state.turn_color == PlayerColor.RED else -1


def repetition_avoidance(state):
    # Penalise positions whose hash already appears in the play history. The
    # side leading on tokens loses (heads to a draw), so eval is reduced from
    # red's perspective when red is ahead and increased when blue is ahead.
    count = state.play_history.get(int(state.zobrist_hash), 0)
    if count == 0:
        return 0
    diff = state.red_tokens - state.blue_tokens
    if diff > 0:
        return -count
    if diff < 0:
        return count
    return 0


def endgame_emphasis(state):
    # Past play_ply 280 the 300-turn timer makes raw token count decisive.
    # Linearly ramp an extra token-diff bonus to 1.0 at ply 300.
    if state.play_ply <= ENDGAME_RAMP_PLY:
        return 0.0
    boost = min(1.0, (state.play_ply - ENDGAME_RAMP_PLY) / ENDGAME_RAMP_LEN)
    return boost * (state.red_tokens - state.blue_tokens)
