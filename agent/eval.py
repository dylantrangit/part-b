TOKEN_DIFF_WEIGHT = 100
STACK_DIFF_WEIGHT = 20
WEIGHTED_HEIGHT_WEIGHT = 15


def evaluate(state):
    score = 0
    score += TOKEN_DIFF_WEIGHT * token_diff(state)
    score += STACK_DIFF_WEIGHT * stack_diff(state)
    score += WEIGHTED_HEIGHT_WEIGHT * weighted_height_diff(state)
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