from referee.game import (
    Coord,
    Direction,
    PlaceAction,
    MoveAction,
    EatAction,
    CascadeAction,
    CellState,
    CARDINAL_DIRECTIONS,
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

    for direction in CARDINAL_DIRECTIONS:
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
    actions = []
    player = state.player_to_move

    # go through the cells, skip if wrong color
    for coord, cell in state.board.items():
        if cell.color != player:
            continue

        for direction in CARDINAL_DIRECTIONS:
            if can_move(state, coord, direction):
                actions.append(MoveAction(coord, direction))

            elif can_eat(state, coord, direction):
                actions.append(EatAction(coord, direction))

            if can_cascade(state, coord, direction):
                actions.append(CascadeAction(coord, direction))

    return actions


def can_move(state, coord, direction):
    from_cell = state.board.get(coord, CellState())

    if from_cell.color != state.player_to_move:
        return False

    try:
        target_coord = coord + direction
    except ValueError:
        return False

    target_cell = state.board.get(target_coord, CellState())

    if target_cell.is_empty or target_cell.color == state.player_to_move:
        return True

    return False


def can_eat(state, coord, direction):
    from_cell = state.board.get(coord, CellState())

    if from_cell.color != state.player_to_move:
        return False

    try:
        target_coord = coord + direction
    except ValueError:
        return False

    target_cell = state.board.get(target_coord, CellState())

    if (
        target_cell.color == state.get_opponent()
        and from_cell.height >= target_cell.height
    ):
        return True

    return False


def can_cascade(state, coord, direction):
    pass


