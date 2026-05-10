from math import inf

from ..core.eval import evaluate
from ..core.time_budget import SearchTimeout
from ..core.ordering import order_moves
from ..core.tt import EXACT, LOWER, UPPER, TTEntry


def signed_eval(board):
    value = evaluate(board)
    return value if board.turn_color.name == "RED" else -value


def negamax_fixed(board, depth, alpha, beta, ply=0):
    t = board.terminal(ply)
    if t is not None:
        return (t if board.turn_color.name == "RED" else -t), None

    if depth == 0:
        return signed_eval(board), None

    best_move = None
    best_val = -inf

    for move in list(board.legal_actions()):
        board.apply(move)
        val, _ = negamax_fixed(board, depth - 1, -beta, -alpha, ply + 1)
        val = -val
        board.undo()

        if val > best_val:
            best_val = val
            best_move = move

        if val > alpha:
            alpha = val

        if alpha >= beta:
            break

    return best_val, best_move


def iterative_deepening(board, budget, tt, max_depth=64):
    best_move = None
    depth = 1
    while depth <= max_depth:
        try:
            _, move = negamax_ab(board, depth, -inf, inf, 0, budget, tt)
        except SearchTimeout:
            break
        if move is not None:
            best_move = move
        if budget.remaining() < budget.per_move_slice():
            break
        depth += 1
    return best_move


def negamax_ab(board, depth, alpha, beta, ply, budget, tt):
    if budget.expired():
        raise SearchTimeout

    t = board.terminal(ply)
    if t is not None:
        return (t if board.turn_color.name == "RED" else -t), None

    # Soft draw cutoff: stop the search chasing repetitions. The hard rule
    # (count >= 3 = draw) is already in board.terminal(); this is a search
    # heuristic to avoid wasting effort on repetition lines.
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
        return signed_eval(board), None

    actions = list(board.legal_actions())
    if not actions:
        return 0, None

    best_val = -inf
    best_move = None
    alpha0 = alpha

    for move in order_moves(board, actions, tt_move):
        board.apply(move)
        try:
            val, _ = negamax_ab(board, depth - 1, -beta, -alpha, ply + 1, budget, tt)
        finally:
            board.undo()
        val = -val

        if val > best_val:
            best_val = val
            best_move = move
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break

    if best_val >= beta:
        flag = LOWER
    elif best_val <= alpha0:
        flag = UPPER
    else:
        flag = EXACT

    tt.put(key, TTEntry(depth, best_val, flag, best_move, age=board.play_ply))

    return best_val, best_move
