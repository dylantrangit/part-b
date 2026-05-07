"""Heavy rollout policy for I5 MCTS.

Picks an action via:
  1. Immediate-win check, gated on opponent having one stack left. Cheap
     scan over our EAT actions for one whose target is the lone enemy stack;
     no apply/undo.
  2. Defensive filter, gated on us having one stack left. Each candidate is
     applied; opponent threat is checked by an O(neighbours) adjacency scan
     instead of a nested opponent-action enumeration. Catches EAT threats
     only — CASCADE-off-board elimination is rare in this regime and not
     handled.
  3. Softmax over a one-pass heuristic: MVV-LVA for EAT, pushed-mass for
     CASCADE, centre proximity for MOVE.
"""
import math
import random

from referee.game import EatAction, CascadeAction, MoveAction, PlayerColor

from .board import DIR_TO_DR_DC


_SOFTMAX_TEMP = 1.0


def rollout_policy_action(state, cached_actions=None, cached_scores=None):
    """Pick a rollout action.

    If `cached_actions` and `cached_scores` are provided, they MUST be the
    full legal-action list for `state` and matching heuristic scores. Used
    by MCTS to skip recomputation on the first rollout step out of a leaf
    whose priors were already materialised at expansion. Cached scores are
    only consumed when the gates don't fire (full-list softmax path);
    otherwise we fall back to fresh computation on the filtered subset.
    """
    actions = cached_actions if cached_actions is not None else list(state.legal_actions())
    if not actions:
        return None

    is_red = state.turn_color.name == "RED"
    enemy_stacks = state.blue_stacks if is_red else state.red_stacks
    own_stacks = state.red_stacks if is_red else state.blue_stacks

    if enemy_stacks <= 1:
        winning = _find_immediate_win(state, actions, is_red)
        if winning is not None:
            return winning

    if own_stacks <= 1:
        safe = _find_safe_actions(state, actions, is_red)
        # Only divert if defence is both needed (some unsafe candidates exist)
        # and possible (some safe candidates exist). Otherwise fall through.
        if 0 < len(safe) < len(actions):
            return _sample_softmax(state, safe)

    return _sample_softmax(state, actions, cached_scores)


def _find_immediate_win(state, actions, is_red):
    """Return an EAT action that eliminates the opponent's last stack, or None.

    Caller has gated on enemy_stacks <= 1, so there's at most one target stack.
    An EAT lands on it iff `coord + direction` equals its position. Heights
    are already validated by legal-action generation (EAT requires h_us >=
    h_them), so any matching EAT is a winning move.
    """
    enemy_pieces = state.blue_pieces if is_red else state.red_pieces
    if not enemy_pieces:
        return None
    target_flat = enemy_pieces[0]
    target_r, target_c = target_flat // 8, target_flat % 8

    for action in actions:
        if not isinstance(action, EatAction):
            continue
        target = action.coord + action.direction
        if target.r == target_r and target.c == target_c:
            return action
    return None


def _find_safe_actions(state, actions, is_red):
    safe = []
    for action in actions:
        state.apply(action)
        threatened = _opponent_can_eat_us(state, is_red)
        state.undo()
        if not threatened:
            safe.append(action)
    return safe


def _opponent_can_eat_us(state, is_red):
    """Adjacency EAT-threat check from the post-our-move state.

    Returns True iff opponent has any stack cardinally adjacent to one of our
    remaining stacks with h_opp >= h_us. Bounded to own_stacks <= 1: with two
    or more stacks, a single EAT cannot eliminate us. Misses cascade-push
    threats; that gap is acceptable in the gated regime.
    """
    own_stacks = state.red_stacks if is_red else state.blue_stacks
    if own_stacks == 0:
        return True
    if own_stacks > 1:
        return False

    own_pieces = state.red_pieces if is_red else state.blue_pieces
    flat = own_pieces[0]
    r, c = flat // 8, flat % 8
    our_h = abs(int(state.grid[r, c]))
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 8 and 0 <= nc < 8):
            continue
        v = int(state.grid[nr, nc])
        if v == 0:
            continue
        is_enemy = (v < 0) if is_red else (v > 0)
        if is_enemy and abs(v) >= our_h:
            return True
    return False


def _sample_softmax(state, actions, scores=None):
    if scores is None:
        scores = [heuristic_score(state, a) for a in actions]
    max_s = max(scores)
    weights = [math.exp((s - max_s) / _SOFTMAX_TEMP) for s in scores]
    total = sum(weights)
    if total <= 0.0:
        return random.choice(actions)
    target = random.random() * total
    cum = 0.0
    for w, action in zip(weights, actions):
        cum += w
        if target <= cum:
            return action
    return actions[-1]


def heuristic_score(state, action):
    grid = state.grid
    if isinstance(action, EatAction):
        sr, sc = action.coord.r, action.coord.c
        dr, dc = DIR_TO_DR_DC[action.direction]
        target_h = abs(int(grid[sr + dr, sc + dc]))
        attacker_h = abs(int(grid[sr, sc]))
        # MVV-LVA: prefer eating big enemies with small attackers.
        return 3.0 + target_h - 0.1 * attacker_h
    if isinstance(action, CascadeAction):
        return 1.0 + 0.5 * _cascade_pushed_mass(state, action)
    if isinstance(action, MoveAction):
        sr, sc = action.coord.r, action.coord.c
        dr, dc = DIR_TO_DR_DC[action.direction]
        nr, nc = sr + dr, sc + dc
        if not (0 <= nr < 8 and 0 <= nc < 8):
            return 0.0
        return 0.1 * (7 - abs(nr - 3.5) - abs(nc - 3.5))
    return 0.0


def _cascade_pushed_mass(state, action):
    grid = state.grid
    sr, sc = action.coord.r, action.coord.c
    h = abs(int(grid[sr, sc]))
    dr, dc = DIR_TO_DR_DC[action.direction]
    is_red = state.turn_color == PlayerColor.RED

    score = 0.0
    fell_off = 0
    cr, cc = sr, sc
    for _ in range(h):
        cr += dr
        cc += dc
        if not (0 <= cr < 8 and 0 <= cc < 8):
            fell_off += 1
            continue
        v = int(grid[cr, cc])
        if v == 0:
            continue
        target_h = abs(v)
        is_enemy = (v < 0) if is_red else (v > 0)
        if is_enemy:
            score += target_h
    # Small bonus for cascading off the edge — own tokens lost is bad, but
    # rays that fall off often clear the way for follow-up captures.
    return score + 0.1 * fell_off
