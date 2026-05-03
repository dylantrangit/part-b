# Iteration 3 search: Principal Variation Search + quiescence (with delta
# pruning) + killers + history + aspiration windows + iterative deepening.
#
# Shares the I2 transposition table format and the I3 move ordering
# (`order_moves_pvs`) plus a separate noisy-move generator for quiescence.

from math import inf

from referee.game import EatAction, CascadeAction

from ..core.eval import evaluate, TOKEN_DIFF_WEIGHT
from ..core.time_budget import SearchTimeout
from ..core.ordering import order_moves_pvs, generate_noisy_actions
from ..core.tt import EXACT, LOWER, UPPER, TTEntry


INF = inf
ASPIRATION_DELTA = 50
DELTA_MARGIN = 200
QUIESCENCE_MAX_DEPTH = 6


def signed_eval(board):
    value = evaluate(board)
    return value if board.turn_color.name == "RED" else -value


def iterative_deepening_pvs(board, budget, tt, max_depth=64):
    best_move = None
    prev_val = 0
    killers = {}
    history = {}
    depth = 1
    while depth <= max_depth:
        try:
            if depth >= 3:
                lo = prev_val - ASPIRATION_DELTA
                hi = prev_val + ASPIRATION_DELTA
                val, move = _pvs(board, depth, lo, hi, 0, budget, tt, killers, history)
                if val <= lo:
                    val, move = _pvs(board, depth, -INF, hi, 0, budget, tt, killers, history)
                elif val >= hi:
                    val, move = _pvs(board, depth, lo, INF, 0, budget, tt, killers, history)
            else:
                val, move = _pvs(board, depth, -INF, INF, 0, budget, tt, killers, history)
        except SearchTimeout:
            break
        if move is not None:
            best_move = move
            prev_val = val
        if budget.remaining() < budget.per_move_slice():
            break
        depth += 1
    return best_move


def _pvs(board, depth, alpha, beta, ply, budget, tt, killers, history):
    if budget.expired():
        raise SearchTimeout

    t = board.terminal(ply)
    if t is not None:
        return (t if board.turn_color.name == "RED" else -t), None

    # Repetition heuristic: count >= 2 means another visit makes it threefold.
    if board.play_history.get(int(board.zobrist_hash), 0) >= 2:
        return 0, None

    key = board.to_key()
    tt_entry = tt.get(key)
    tt_move = tt_entry.best_move if tt_entry is not None else None

    if tt_entry is not None and tt_entry.depth >= depth:
        v = tt_entry.value
        f = tt_entry.flag
        if f == EXACT:
            return v, tt_entry.best_move
        if f == LOWER and v > alpha:
            alpha = v
        elif f == UPPER and v < beta:
            beta = v
        if alpha >= beta:
            return v, tt_entry.best_move

    if depth == 0:
        return _quiescence(board, alpha, beta, ply, QUIESCENCE_MAX_DEPTH, budget), None

    actions = list(board.legal_actions())
    if not actions:
        return 0, None

    killers_at_ply = killers.get(ply, ())
    ordered = order_moves_pvs(board, actions, tt_move, killers_at_ply, history)

    best_val = -INF
    best_move = None
    alpha0 = alpha

    for i, move in enumerate(ordered):
        board.apply(move)
        try:
            if i == 0:
                val, _ = _pvs(board, depth - 1, -beta, -alpha, ply + 1, budget, tt, killers, history)
                val = -val
            else:
                # Null-window scout
                val, _ = _pvs(board, depth - 1, -alpha - 1, -alpha, ply + 1, budget, tt, killers, history)
                val = -val
                if alpha < val < beta:
                    # Fail-soft re-search with the full window
                    val, _ = _pvs(board, depth - 1, -beta, -alpha, ply + 1, budget, tt, killers, history)
                    val = -val
        finally:
            board.undo()

        if val > best_val:
            best_val = val
            best_move = move
        if val > alpha:
            alpha = val
        if alpha >= beta:
            if not isinstance(move, EatAction):
                _store_killer(killers, ply, move)
                history[move] = history.get(move, 0) + depth * depth
            break

    if best_val >= beta:
        flag = LOWER
    elif best_val <= alpha0:
        flag = UPPER
    else:
        flag = EXACT

    tt.put(key, TTEntry(depth, best_val, flag, best_move, age=board.play_ply))

    return best_val, best_move


def _quiescence(board, alpha, beta, ply, q_depth, budget):
    if budget.expired():
        raise SearchTimeout

    t = board.terminal(ply)
    if t is not None:
        return t if board.turn_color.name == "RED" else -t

    stand_pat = signed_eval(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    if q_depth == 0:
        return alpha

    noisy = list(generate_noisy_actions(board))
    if not noisy:
        return alpha

    eats = [a for a in noisy if isinstance(a, EatAction)]
    cascades = [a for a in noisy if isinstance(a, CascadeAction)]
    eats.sort(
        key=lambda a: -(
            100 * board.get_height(a.coord + a.direction) - board.get_height(a.coord)
        ),
    )

    for move in eats + cascades:
        if isinstance(move, EatAction):
            target_h = board.get_height(move.coord + move.direction)
            gain = TOKEN_DIFF_WEIGHT * target_h
            if stand_pat + gain + DELTA_MARGIN < alpha:
                continue

        board.apply(move)
        try:
            val = -_quiescence(board, -beta, -alpha, ply + 1, q_depth - 1, budget)
        finally:
            board.undo()

        if val >= beta:
            return beta
        if val > alpha:
            alpha = val

    return alpha


def _store_killer(killers, ply, move):
    bucket = killers.get(ply)
    if bucket is None:
        killers[ply] = [move]
        return
    if bucket[0] == move:
        return
    if len(bucket) == 1:
        killers[ply] = [move, bucket[0]]
    else:
        killers[ply] = [move, bucket[0]]
