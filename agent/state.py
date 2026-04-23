import copy

from referee.game import PlayerColor, Coord, CellState


class GameState:

    def __init__(self):
        # only store non empty cell. ei. board = cell here and here. empty cells are implicit
        self.board = {}  # dict[Coord, CellState]

        # red always starts first
        self.player_to_move = PlayerColor.RED
        #AFTer 8 we go into gameplay
        self.placement_count = 0
        self.play_phase_turn_count = 0

        # For repetition tracking later
        self.position_history = []

    def get_phase(self):
        if self.placement_count < 8:
            return "placement"
        return "play"

    def copy(self):
        #so that we can copy game states for search algos to search forward
        new_state = GameState()
        new_state.board = copy.deepcopy(self.board)
        new_state.player_to_move = self.player_to_move
        new_state.placement_count = self.placement_count
        new_state.play_phase_turn_count = self.play_phase_turn_count
        new_state.position_history = self.position_history.copy()
        return new_state


    def in_bounds(self, coord):
        return 0 <= coord.r < 8 and 0 <= coord.c < 8

    def to_key(self):
        cells = []
        for coord, cell in self.board.items():
            if cell.is_stack:
                cells.append((coord.r, coord.c, cell.color, cell.height))

        cells.sort()

        return (
            tuple(cells),
            self.player_to_move,
            self.placement_count,
            self.play_phase_turn_count
        )