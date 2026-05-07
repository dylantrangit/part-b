from referee.game import EatAction, CascadeAction, PlayerColor


def order_moves(state, actions, tt_move=None):
    tt_first = []
    eats = []
    pushing_cascades = []
    others = []

    for action in actions:
        if tt_move is not None and action == tt_move:
            tt_first.append(action)
            continue
        if isinstance(action, EatAction):
            eats.append(action)
            continue
        if isinstance(action, CascadeAction) and _cascade_pushes_off(state, action):
            pushing_cascades.append(action)
            continue
        others.append(action)

    eats.sort(key=lambda a: -_mvv_lva(state, a))
    return tt_first + eats + pushing_cascades + others


def order_moves_pvs(state, actions, tt_move, killers_at_ply, history):
    # I3 ordering: TT move, EATs (MVV-LVA), noisy cascades, killers, then
    # remaining quiet moves sorted by history score.
    tt_first = []
    eats = []
    noisy_cascades = []
    killer_hits = []
    quiet = []

    killer_set = set(killers_at_ply or ())

    for action in actions:
        if tt_move is not None and action == tt_move:
            tt_first.append(action)
            continue
        if isinstance(action, EatAction):
            eats.append(action)
            continue
        if isinstance(action, CascadeAction) and _cascade_is_noisy(state, action):
            noisy_cascades.append(action)
            continue
        if action in killer_set:
            killer_hits.append(action)
            continue
        quiet.append(action)

    eats.sort(key=lambda a: -_mvv_lva(state, a))
    quiet.sort(key=lambda a: -history.get(a, 0))
    return tt_first + eats + noisy_cascades + killer_hits + quiet


def generate_noisy_actions(state):
    # Quiescence move set: every EAT plus every CASCADE whose ray either
    # touches an enemy stack or runs off the board edge.
    for action in state.legal_actions():
        if isinstance(action, EatAction):
            yield action
        elif isinstance(action, CascadeAction) and _cascade_is_noisy(state, action):
            yield action


def _mvv_lva(state, action):
    target = action.coord + action.direction
    target_h = state.get_height(target)
    attacker_h = state.get_height(action.coord)
    return 100 * target_h - attacker_h


def _cascade_pushes_off(state, action):
    coord = action.coord
    h = state.get_height(coord)
    direction = action.direction
    turn_is_red = state.turn_color == PlayerColor.RED

    enemy_on_ray = False
    fell_off = False
    cell = coord
    for _ in range(h):
        try:
            cell = cell + direction
        except ValueError:
            fell_off = True
            break
        if not state.in_bounds(cell):
            fell_off = True
            break
        v = state.get_value(cell)
        if v != 0:
            is_enemy = (v < 0) if turn_is_red else (v > 0)
            if is_enemy:
                enemy_on_ray = True

    return enemy_on_ray and fell_off


def _cascade_is_noisy(state, action):
    coord = action.coord
    h = state.get_height(coord)
    direction = action.direction
    turn_is_red = state.turn_color == PlayerColor.RED

    cell = coord
    for _ in range(h):
        try:
            cell = cell + direction
        except ValueError:
            return True
        if not state.in_bounds(cell):
            return True
        v = state.get_value(cell)
        if v != 0:
            is_enemy = (v < 0) if turn_is_red else (v > 0)
            if is_enemy:
                return True
    return False
