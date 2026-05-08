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


# Two-temperature setup (Tier 2.5):
# - τ_rollout (broader): preserves playout diversity so MCTS sees varied
#   futures from each leaf. The original 1.0 was too greedy (F3 fix).
# - τ_prior (sharper): peakier PUCT priors concentrate visits on plausible
#   moves earlier — the diagnostic showed opening Q-noise dominates with
#   only ~15 sims/child, so sharper priors directly help PUCT focus.
_SOFTMAX_TEMP = 2.0  # τ_rollout — used in _sample_softmax
_TAU_PRIOR = 1.0     # used by _materialise_actions in mcts_heavy


def rollout_policy_action(state, cached_actions=None, cached_priors=None):
    """Pick a rollout action.

    If `cached_actions` and `cached_priors` are provided, they MUST be the
    full legal-action list for `state` and the matching softmax-normalised
    priors. Used by MCTS to skip recomputation on the first rollout step
    out of a leaf whose priors were materialised at expansion. The cache
    is only consumed when the gates don't fire (full-list softmax path);
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

    return _sample_softmax(state, actions, cached_priors)


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


def _sample_softmax(state, actions, weights=None):
    """Sample one action proportionally to `weights`.

    If `weights` is provided, treat them as direct sampling weights — they
    do not need to sum to 1 (re-normalised here). Pass cached softmax-normalised
    priors via this argument to skip recomputation.

    If omitted, compute fresh `exp(heuristic / τ)` weights.
    """
    if weights is None:
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
    is_red = state.turn_color == PlayerColor.RED
    if isinstance(action, EatAction):
        sr, sc = action.coord.r, action.coord.c
        dr, dc = DIR_TO_DR_DC[action.direction]
        tr, tc = sr + dr, sc + dc
        target_h = abs(int(grid[tr, tc]))
        attacker_h = abs(int(grid[sr, sc]))
        # MVV-LVA base: prefer eating big enemies with small attackers.
        score = 3.0 + target_h - 0.1 * attacker_h
        # Tier 3.3: post-move safety. After the EAT, the attacker sits at
        # (tr, tc) at its original height. If an enemy adjacent to (tr, tc)
        # has h_enemy >= attacker_h, opponent recaptures next turn for free.
        # Diagnostic showed Red-side fast tactical losses are exactly this
        # pattern (greedy EAT → free recapture).
        for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ar, ac = tr + ddr, tc + ddc
            if not (0 <= ar < 8 and 0 <= ac < 8):
                continue
            adj = int(grid[ar, ac])
            if adj == 0:
                continue
            # The just-eaten square's pre-move adjacency: skip the source we
            # came from since after EAT it'll be empty (we left it).
            if (ar, ac) == (sr, sc):
                continue
            if ((adj < 0) if is_red else (adj > 0)) and abs(adj) >= attacker_h:
                score -= 0.8
                break
        return score
    if isinstance(action, CascadeAction):
        sr, sc = action.coord.r, action.coord.c
        h = abs(int(grid[sr, sc]))
        dr, dc = DIR_TO_DR_DC[action.direction]
        enemy_mass, enemy_off_board, friendly_mass, fell_off = _cascade_quality(
            grid, sr, sc, h, dr, dc, is_red
        )
        # Cascading converts a height-h stack into h height-1 tokens.
        # Productive only when enough enemy mass is hit to offset the loss
        # of structure. The trailing -0.3 (F7) is an extra flat down-weight
        # so cascades have to clear a bar before competing with MOVE/EAT.
        # Tier 3.4: extra bonus when an enemy is pushed off the edge —
        # eliminated tokens are strictly better than relocated ones.
        return (
            0.6 * enemy_mass         # productive: push enemies (relocate or off)
            + 0.4 * enemy_off_board  # bonus: pushed-off enemies are eliminated
            - 0.3 * h                # structural cost: tall stack → singletons
            - 0.4 * friendly_mass    # disrupting own pieces in the path
            - 0.5 * fell_off         # own cascading tokens lost off the edge
            - 0.3                    # F7: flat CASCADE bias
        )
    if isinstance(action, MoveAction):
        sr, sc = action.coord.r, action.coord.c
        dr, dc = DIR_TO_DR_DC[action.direction]
        nr, nc = sr + dr, sc + dc
        if not (0 <= nr < 8 and 0 <= nc < 8):
            return 0.0
        return _move_score(grid, sr, sc, nr, nc, is_red)
    return 0.0


def _cascade_quality(grid, sr, sc, h, dr, dc, is_red):
    """Walk the cascade ray; return (enemy_mass, enemy_off_board, friendly_mass, fell_off).

    `enemy_mass` and `friendly_mass` sum the heights of stacks the cascading
    tokens would push (per the spec, pushed not merged). `enemy_off_board`
    is the subset of enemy mass whose one-step push lands off the board
    (eliminated). `fell_off` counts cascade-token positions past the board
    edge — those own tokens are lost.

    The off-board check is approximate: only one-step pushes are simulated
    so chain-pushes that drive an enemy off via multiple relocations aren't
    detected. Acceptable for a rollout heuristic.
    """
    enemy_mass = 0
    enemy_off_board = 0
    friendly_mass = 0
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
        if (v < 0) if is_red else (v > 0):
            enemy_mass += target_h
            # If the push destination is off-board, this enemy is eliminated.
            push_r, push_c = cr + dr, cc + dc
            if not (0 <= push_r < 8 and 0 <= push_c < 8):
                enemy_off_board += target_h
        else:
            friendly_mass += target_h
    return enemy_mass, enemy_off_board, friendly_mass, fell_off


def _move_score(grid, sr, sc, nr, nc, is_red):
    """Heuristic for a MOVE: centre tiebreak + merge bonus + threat relief +
    threat creation + edge penalty. One neighbour walk per (source, dest)."""
    h = abs(int(grid[sr, sc]))
    dest_v = int(grid[nr, nc])

    # Centre proximity as a small tiebreaker; range ~0..0.7.
    score = 0.1 * (7 - abs(nr - 3.5) - abs(nc - 3.5))

    # Tier 3.5: explicit penalty for moving onto an edge cell — pieces
    # there have ≤3 escape directions and are easier to corner.
    if nr == 0 or nr == 7 or nc == 0 or nc == 7:
        score -= 0.3

    # Merge: dest holds a friendly stack — consolidate. Larger merge target
    # → bigger gain (single tall stack is harder to eat).
    if dest_v != 0:
        # Legal-action gen guarantees dest is empty or friendly here.
        score += 0.5 * abs(dest_v)

    # Threat relief: source is currently threatened by an adjacent stronger
    # or equal enemy → leaving the square removes the threat.
    for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ar, ac = sr + ddr, sc + ddc
        if not (0 <= ar < 8 and 0 <= ac < 8):
            continue
        adj = int(grid[ar, ac])
        if adj == 0:
            continue
        if ((adj < 0) if is_red else (adj > 0)) and abs(adj) >= h:
            score += 0.5
            break

    # Dest neighbourhood: per-direction bonus for weaker enemies (we can EAT
    # them with priority next turn), penalty for equal-or-stronger enemies
    # (opponent moves first and can EAT us). Mutual-threat squares are bad
    # for the side that just moved — hence the strict `>` for our bonus.
    we_threaten = False
    they_threaten_us = False
    for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ar, ac = nr + ddr, nc + ddc
        if not (0 <= ar < 8 and 0 <= ac < 8):
            continue
        adj = int(grid[ar, ac])
        if adj == 0:
            continue
        if (adj < 0) if is_red else (adj > 0):
            adj_h = abs(adj)
            if h > adj_h:
                we_threaten = True
            else:  # adj_h >= h
                they_threaten_us = True
    if we_threaten and not they_threaten_us:
        score += 0.3
    if they_threaten_us:
        score -= 0.5

    return score
