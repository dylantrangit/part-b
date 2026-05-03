# Iteration 1 search: fixed-depth negamax alpha-beta. No iterative deepening,
# no transposition table, no move ordering.

from math import inf

from ..core.eval import evaluate


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
