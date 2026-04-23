from referee.game import (
    Coord,
    Direction,
    PlaceAction,
    MoveAction,
    EatAction,
    CascadeAction,
    CellState,
)

from .state import GameState


def generate_legal_actions(state):
    if state.get_phase() == "placement":
        return generate_place_actions(state)
    return generate_play_actions(state)


def generate_place_actions(state):
    actions = []

    for r in range(8):
        for c in range(8):
            coord = Coord(r, c)
            if is_valid_placement(state, coord):
                actions.append(PlaceAction(coord))

    return actions


def is_valid_placement(state, coord):
    # must be on board
    if not state.in_bounds(coord):
        return False

    # must be empty
    if not state.board.get(coord, CellState()).is_empty:
        return False

    # after the first placement of the game, cannot place adjacent to opponent
    if state.placement_count > 0 and is_adjacent_to_opponent(state, coord):
        return False

    return True


def is_adjacent_to_opponent(state, coord):
    opponent = state.get_opponent()

    for direction in Direction:
        try:
            next_coord = coord + direction
        except ValueError:
            continue

        if not state.in_bounds(next_coord):
            continue

        cell = state.board.get(next_coord, CellState())
        if cell.color == opponent:
            return True

    return False


def generate_play_actions(state):
    # can jsut reuse the part a code i think like move, eat, cascade
    pass


