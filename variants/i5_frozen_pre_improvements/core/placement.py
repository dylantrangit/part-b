import traceback
from math import inf

from referee.game import Coord, PlayerColor

from .eval import evaluate


INTERIOR_WEIGHT = 8
SPREAD_PENALTY = 6
THREAT_BONUS = 10


def choose_placement_action(state):
    try:
        best_action = _search_best_placement(state)
    except Exception:
        traceback.print_exc()
        best_action = None

    if best_action is not None:
        return best_action

    for action in state.legal_actions():
        return action
    return None


def _search_best_placement(state):
    actions = list(state.legal_actions())
    if not actions:
        return None

    maximizing = state.turn_color == PlayerColor.RED
    best_action = actions[0]
    best_score = -inf if maximizing else inf

    alpha = -inf
    beta = inf

    for action in order_place_actions(state, actions):
        state.apply(action)
        try:
            score = placement_search(state, depth=1, alpha=alpha, beta=beta)
        finally:
            state.undo()

        if maximizing:
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score = score
                best_action = action
            beta = min(beta, best_score)

    return best_action


def placement_search(state, depth, alpha, beta):
    if depth == 0 or state.get_phase() != "placement":
        return evaluate(state) + placement_bonus(state)

    actions = list(state.legal_actions())
    if not actions:
        return evaluate(state) + placement_bonus(state)

    maximizing = state.turn_color == PlayerColor.RED

    if maximizing:
        value = -inf
        for action in order_place_actions(state, actions):
            state.apply(action)
            try:
                value = max(value, placement_search(state, depth - 1, alpha, beta))
            finally:
                state.undo()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = inf
    for action in order_place_actions(state, actions):
        state.apply(action)
        try:
            value = min(value, placement_search(state, depth - 1, alpha, beta))
        finally:
            state.undo()
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def placement_bonus(state):
    red_score = 0
    for flat_idx in state.red_pieces:
        coord = state.coord_from_flat(flat_idx)
        red_score += placement_bonus_for_piece(state, coord, PlayerColor.RED)

    blue_score = 0
    for flat_idx in state.blue_pieces:
        coord = state.coord_from_flat(flat_idx)
        blue_score += placement_bonus_for_piece(state, coord, PlayerColor.BLUE)

    return red_score - blue_score


def placement_bonus_for_piece(state, coord, colour):
    score = 0
    score += INTERIOR_WEIGHT * interior_value(coord.r, coord.c)
    score -= SPREAD_PENALTY * count_adjacent_colour(state, coord, colour)

    enemy = PlayerColor.BLUE if colour == PlayerColor.RED else PlayerColor.RED
    score += THREAT_BONUS * count_adjacent_colour(state, coord, enemy)

    return score


def interior_value(r, c):
    return 7 - (abs(r - 3.5) + abs(c - 3.5))


def count_adjacent_colour(state, coord, colour):
    count = 0

    for delta_r, delta_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_r = coord.r + delta_r
        new_c = coord.c + delta_c

        if not (0 <= new_r < 8 and 0 <= new_c < 8):
            continue

        nxt = Coord(new_r, new_c)

        if state.get_colour(nxt) == colour:
            count += 1

    return count


def order_place_actions(state, actions):
    scored = []

    for action in actions:
        coord = action.coord
        score = INTERIOR_WEIGHT * interior_value(coord.r, coord.c)

        own_adj = count_adjacent_colour(state, coord, state.turn_color)
        enemy_adj = count_adjacent_colour(state, coord, state.get_opponent())

        score -= SPREAD_PENALTY * own_adj
        score += THREAT_BONUS * enemy_adj

        scored.append((score, action))

    reverse = state.turn_color == PlayerColor.RED
    scored.sort(key=lambda x: x[0], reverse=reverse)
    return [action for _, action in scored]


# from math import inf

# from referee.game import Coord, PlaceAction, PlayerColor

# from .board import generate_place_actions
# from .eval import evaluate


# INTERIOR_WEIGHT = 8
# SPREAD_PENALTY = 6
# THREAT_BONUS = 10


# RED_BOOK = {
#     0: Coord(3, 3),
#     2: Coord(4, 4),
#     4: Coord(3, 6),
#     6: Coord(4, 1),
# }


# def choose_placement_action(state):
#     book_action = get_book_action(state)
#     if book_action is not None:
#         return book_action

#     actions = list(generate_place_actions(state))
#     if not actions:
#         return None

#     maximizing = state.turn_color == PlayerColor.RED

#     best_action = actions[0]
#     best_score = -inf if maximizing else inf

#     alpha = -inf
#     beta = inf

#     for action in order_place_actions(state, actions):
#         child = copy_state_for_placement(state)
#         apply_place_action(child, action)

#         score = placement_search(child, depth=1, alpha=alpha, beta=beta)

#         if maximizing:
#             if score > best_score:
#                 best_score = score
#                 best_action = action
#             alpha = max(alpha, best_score)
#         else:
#             if score < best_score:
#                 best_score = score
#                 best_action = action
#             beta = min(beta, best_score)

#     return best_action


# def placement_search(state, depth, alpha, beta):
#     if depth == 0 or state.get_phase() != "placement":
#         return evaluate(state) + placement_bonus(state)

#     actions = list(generate_place_actions(state))
#     if not actions:
#         return evaluate(state) + placement_bonus(state)

#     maximizing = state.turn_color == PlayerColor.RED

#     if maximizing:
#         value = -inf
#         for action in order_place_actions(state, actions):
#             child = copy_state_for_placement(state)
#             apply_place_action(child, action)

#             value = max(value, placement_search(child, depth - 1, alpha, beta))
#             alpha = max(alpha, value)
#             if alpha >= beta:
#                 break
#         return value

#     value = inf
#     for action in order_place_actions(state, actions):
#         child = copy_state_for_placement(state)
#         apply_place_action(child, action)

#         value = min(value, placement_search(child, depth - 1, alpha, beta))
#         beta = min(beta, value)
#         if alpha >= beta:
#             break
#     return value


# def placement_bonus(state):
#     red_score = 0
#     for flat_idx in state.red_pieces:
#         coord = state.coord_from_flat(flat_idx)
#         red_score += placement_bonus_for_piece(state, coord, PlayerColor.RED)

#     blue_score = 0
#     for flat_idx in state.blue_pieces:
#         coord = state.coord_from_flat(flat_idx)
#         blue_score += placement_bonus_for_piece(state, coord, PlayerColor.BLUE)

#     return red_score - blue_score


# def placement_bonus_for_piece(state, coord, colour):
#     score = 0
#     score += INTERIOR_WEIGHT * interior_value(coord.r, coord.c)
#     score -= SPREAD_PENALTY * count_adjacent_colour(state, coord, colour)

#     enemy = PlayerColor.BLUE if colour == PlayerColor.RED else PlayerColor.RED
#     score += THREAT_BONUS * count_adjacent_colour(state, coord, enemy)

#     return score


# def interior_value(r, c):
#     return 7 - (abs(r - 3.5) + abs(c - 3.5))


# # def count_adjacent_colour(state, coord, colour):
# #     count = 0

# #     for delta_r, delta_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
# #         nxt = Coord(coord.r + delta_r, coord.c + delta_c)

# #         if not state.in_bounds(nxt):
# #             continue

# #         if state.get_colour(nxt) == colour:
# #             count += 1

# #     return count

# def count_adjacent_colour(state, coord, colour):
#     count = 0

#     for delta_r, delta_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#         new_r = coord.r + delta_r
#         new_c = coord.c + delta_c

#         if not (0 <= new_r < 8 and 0 <= new_c < 8):
#             continue

#         nxt = Coord(new_r, new_c)

#         if state.get_colour(nxt) == colour:
#             count += 1

#     return count


# def order_place_actions(state, actions):
#     scored = []

#     for action in actions:
#         coord = action.coord
#         score = INTERIOR_WEIGHT * interior_value(coord.r, coord.c)

#         own_adj = count_adjacent_colour(state, coord, state.turn_color)
#         enemy_adj = count_adjacent_colour(state, coord, state.get_opponent())

#         score -= SPREAD_PENALTY * own_adj
#         score += THREAT_BONUS * enemy_adj

#         scored.append((score, action))

#     reverse = state.turn_color == PlayerColor.RED
#     scored.sort(key=lambda x: x[0], reverse=reverse)
#     return [action for _, action in scored]


# def get_book_action(state):
#     if state.turn_color != PlayerColor.RED:
#         return None

#     coord = RED_BOOK.get(state.placement_count)
#     if coord is None:
#         return None

#     for action in generate_place_actions(state):
#         if action.coord == coord:
#             return PlaceAction(coord)

#     return None


# def apply_place_action(state, action):
#     value = 1 if state.turn_color == PlayerColor.RED else -1
#     state.set_cell(action.coord, value)
#     state.placement_count += 1
#     state.switch_turn()


# def copy_state_for_placement(state):
#     from .board import GameState

#     new_state = GameState()
#     new_state.grid = state.grid.copy()

#     new_state.red_pieces = state.red_pieces.copy()
#     new_state.blue_pieces = state.blue_pieces.copy()
#     new_state._red_piece_pos = state._red_piece_pos.copy()
#     new_state._blue_piece_pos = state._blue_piece_pos.copy()

#     new_state.red_tokens = state.red_tokens
#     new_state.blue_tokens = state.blue_tokens
#     new_state.red_stacks = state.red_stacks
#     new_state.blue_stacks = state.blue_stacks
#     new_state.placement_count = state.placement_count
#     new_state.turn_color = state.turn_color
#     new_state.play_ply = state.play_ply
#     new_state.zobrist_hash = state.zobrist_hash
#     new_state.play_history = state.play_history.copy()
#     new_state.undo_stack = state.undo_stack.copy()

#     return new_state