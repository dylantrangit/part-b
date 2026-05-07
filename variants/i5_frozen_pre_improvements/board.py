import numpy as np

from referee.game import (
    Coord,
    PlaceAction,
    MoveAction,
    EatAction,
    CascadeAction,
    PlayerColor,
    CARDINAL_DIRECTIONS,
)

from .zobrist import Z, SIDE_TO_MOVE_KEY, MAX_HEIGHT_BUCKET


BOARD_SIZE = 8
EMPTY = 0
WIN_VALUE = 10**9


class GameState:
    def __init__(self):
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        self.red_pieces = []
        self.blue_pieces = []

        self._red_piece_pos = {}
        self._blue_piece_pos = {}

        self.red_tokens = 0
        self.blue_tokens = 0
        self.red_stacks = 0
        self.blue_stacks = 0
        self.placement_count = 0
        self.turn_color = PlayerColor.RED
        self.play_ply = 0

        self.zobrist_hash = np.uint64(0)
        self.play_history = {}
        self.undo_stack = []

    def get_phase(self):
        return "placement" if self.placement_count < 8 else "play"

    def get_opponent(self):
        return PlayerColor.BLUE if self.turn_color == PlayerColor.RED else PlayerColor.RED

    def in_bounds(self, coord):
        return 0 <= coord.r < 8 and 0 <= coord.c < 8

    def flat_index(self, coord):
        return coord.r * 8 + coord.c

    def coord_from_flat(self, flat_idx):
        return Coord(flat_idx // 8, flat_idx % 8)

    def get_value(self, coord):
        return int(self.grid[coord.r, coord.c])

    def get_height(self, coord):
        return abs(int(self.grid[coord.r, coord.c]))

    def get_colour(self, coord):
        value = int(self.grid[coord.r, coord.c])
        if value > 0:
            return PlayerColor.RED
        if value < 0:
            return PlayerColor.BLUE
        return None

    def is_empty(self, coord):
        return int(self.grid[coord.r, coord.c]) == 0

    def current_piece_list(self):
        return self.red_pieces if self.turn_color == PlayerColor.RED else self.blue_pieces

    def _zobrist_piece_key(self, flat_idx, value):
        if value == 0:
            return np.uint64(0)

        colour_idx = 0 if value > 0 else 1
        height_bucket = min(abs(value), MAX_HEIGHT_BUCKET)
        return Z[flat_idx, colour_idx, height_bucket]

    def _add_piece_index(self, flat_idx, colour):
        if colour == PlayerColor.RED:
            self._red_piece_pos[flat_idx] = len(self.red_pieces)
            self.red_pieces.append(flat_idx)
        else:
            self._blue_piece_pos[flat_idx] = len(self.blue_pieces)
            self.blue_pieces.append(flat_idx)

    def _remove_piece_index(self, flat_idx, colour):
        if colour == PlayerColor.RED:
            pieces = self.red_pieces
            pos_map = self._red_piece_pos
        else:
            pieces = self.blue_pieces
            pos_map = self._blue_piece_pos

        remove_pos = pos_map[flat_idx]
        last_idx = pieces[-1]

        pieces[remove_pos] = last_idx
        pos_map[last_idx] = remove_pos

        pieces.pop()
        del pos_map[flat_idx]

    def set_cell(self, coord, new_value):
        old_value = self.get_value(coord)
        if old_value == new_value:
            return

        flat_idx = self.flat_index(coord)

        old_colour = None
        if old_value > 0:
            old_colour = PlayerColor.RED
        elif old_value < 0:
            old_colour = PlayerColor.BLUE

        new_colour = None
        if new_value > 0:
            new_colour = PlayerColor.RED
        elif new_value < 0:
            new_colour = PlayerColor.BLUE

        old_height = abs(old_value)
        new_height = abs(new_value)

        if old_value != 0:
            self.zobrist_hash ^= self._zobrist_piece_key(flat_idx, old_value)
            self._remove_piece_index(flat_idx, old_colour)

            if old_colour == PlayerColor.RED:
                self.red_tokens -= old_height
                self.red_stacks -= 1
            else:
                self.blue_tokens -= old_height
                self.blue_stacks -= 1

        self.grid[coord.r, coord.c] = new_value

        if new_value != 0:
            self._add_piece_index(flat_idx, new_colour)
            self.zobrist_hash ^= self._zobrist_piece_key(flat_idx, new_value)

            if new_colour == PlayerColor.RED:
                self.red_tokens += new_height
                self.red_stacks += 1
            else:
                self.blue_tokens += new_height
                self.blue_stacks += 1

    def switch_turn(self):
        self.turn_color = self.get_opponent()
        self.zobrist_hash ^= SIDE_TO_MOVE_KEY

    def record_play_history(self):
        if self.get_phase() != "play":
            return
        key = int(self.zobrist_hash)
        self.play_history[key] = self.play_history.get(key, 0) + 1

    def unrecord_play_history(self):
        if self.get_phase() != "play":
            return
        key = int(self.zobrist_hash)
        count = self.play_history.get(key, 0)
        if count <= 1:
            self.play_history.pop(key, None)
        else:
            self.play_history[key] = count - 1

    def is_threefold_repetition(self):
        return self.play_history.get(int(self.zobrist_hash), 0) >= 3

    def to_key(self):
        return int(self.zobrist_hash)

    def legal_actions(self):
        return generate_legal_actions(self)

    def apply(self, action):
        apply_action(self, action)

    def terminal(self, ply=0):
        if self.red_tokens == 0:
            return -(WIN_VALUE - ply)

        if self.blue_tokens == 0:
            return WIN_VALUE - ply

        if self.is_threefold_repetition():
            return 0

        if not has_any_legal_action(self):
            return 0

        if self.play_ply >= 300:
            if self.red_tokens > self.blue_tokens:
                return WIN_VALUE - ply
            if self.blue_tokens > self.red_tokens:
                return -(WIN_VALUE - ply)
            return 0

        return None

    def copy(self):
        new_state = GameState()

        new_state.grid = self.grid.copy()

        new_state.red_pieces = self.red_pieces.copy()
        new_state.blue_pieces = self.blue_pieces.copy()

        new_state._red_piece_pos = self._red_piece_pos.copy()
        new_state._blue_piece_pos = self._blue_piece_pos.copy()

        new_state.red_tokens = self.red_tokens
        new_state.blue_tokens = self.blue_tokens
        new_state.red_stacks = self.red_stacks
        new_state.blue_stacks = self.blue_stacks

        new_state.placement_count = self.placement_count
        new_state.turn_color = self.turn_color
        new_state.play_ply = self.play_ply

        new_state.zobrist_hash = self.zobrist_hash
        new_state.play_history = self.play_history.copy()
        new_state.undo_stack = self.undo_stack.copy()

        return new_state


def generate_legal_actions(state):
    if state.get_phase() == "placement":
        yield from generate_place_actions(state)
    else:
        yield from generate_play_actions(state)


def generate_place_actions(state):
    for r in range(8):
        for c in range(8):
            coord = Coord(r, c)

            if not state.is_empty(coord):
                continue

            if state.placement_count > 0 and is_adjacent_to_opponent(state, coord):
                continue

            yield PlaceAction(coord)


def is_adjacent_to_opponent(state, coord):
    opponent = state.get_opponent()

    for direction in CARDINAL_DIRECTIONS:
        try:
            next_coord = coord + direction
        except ValueError:
            continue

        if not state.in_bounds(next_coord):
            continue

        if state.get_colour(next_coord) == opponent:
            return True

    return False


def generate_play_actions(state):
    for flat_idx in state.current_piece_list():
        coord = state.coord_from_flat(flat_idx)
        height = state.get_height(coord)

        for direction in CARDINAL_DIRECTIONS:
            try:
                target_coord = coord + direction
                in_bounds = state.in_bounds(target_coord)
            except ValueError:
                in_bounds = False
                target_coord = None

            if in_bounds:
                if state.is_empty(target_coord):
                    yield MoveAction(coord, direction)
                else:
                    target_colour = state.get_colour(target_coord)

                    if target_colour == state.turn_color:
                        yield MoveAction(coord, direction)
                    else:
                        if height >= state.get_height(target_coord):
                            yield EatAction(coord, direction)

            if height >= 2:
                yield CascadeAction(coord, direction)


def has_any_legal_action(state):
    for _ in state.legal_actions():
        return True
    return False


def apply_action(state, action):
    if isinstance(action, PlaceAction):
        _apply_place(state, action)
    elif isinstance(action, MoveAction):
        _apply_move(state, action)
    elif isinstance(action, EatAction):
        _apply_eat(state, action)
    elif isinstance(action, CascadeAction):
        _apply_cascade(state, action)
    else:
        raise ValueError(f"Unknown action: {action}")


def _apply_place(state, action):
    coord = action.coord
    value = 3 if state.turn_color == PlayerColor.RED else -3

    state.set_cell(coord, value)
    state.placement_count += 1
    state.switch_turn()


def _apply_move(state, action):
    from_coord = action.coord
    to_coord = from_coord + action.direction

    from_value = state.get_value(from_coord)
    to_value = state.get_value(to_coord)

    state.set_cell(from_coord, 0)
    state.set_cell(to_coord, from_value + to_value)

    state.play_ply += 1
    state.switch_turn()
    state.record_play_history()


def _apply_eat(state, action):
    from_coord = action.coord
    to_coord = from_coord + action.direction

    from_value = state.get_value(from_coord)

    state.set_cell(from_coord, 0)
    state.set_cell(to_coord, from_value)

    state.play_ply += 1
    state.switch_turn()
    state.record_play_history()


def _step_coord(state, coord, direction, steps=1):
    current = coord
    for _ in range(steps):
        try:
            current = current + direction
        except ValueError:
            return None
        if not state.in_bounds(current):
            return None
    return current


def _push_stack(state, coord, direction):
    value = state.get_value(coord)
    if value == 0:
        return

    next_coord = _step_coord(state, coord, direction)
    if next_coord is None:
        state.set_cell(coord, 0)
        return

    if not state.is_empty(next_coord):
        _push_stack(state, next_coord, direction)

    state.set_cell(coord, 0)
    state.set_cell(next_coord, value)


def _apply_cascade(state, action):
    from_coord = action.coord
    direction = action.direction

    from_value = state.get_value(from_coord)
    height = abs(from_value)
    token_value = 1 if from_value > 0 else -1

    state.set_cell(from_coord, 0)

    for step in range(1, height + 1):
        target_coord = _step_coord(state, from_coord, direction, step)

        if target_coord is None:
            continue

        if not state.is_empty(target_coord):
            _push_stack(state, target_coord, direction)

        state.set_cell(target_coord, token_value)

    state.play_ply += 1
    state.switch_turn()
    state.record_play_history()

