"""
Hand-built tactical positions for the Cascade agent.

Each puzzle is a dict:
    name:        short label
    desc:        what's being tested
    pieces:      {(r, c): signed_height}    positive = red, negative = blue
    turn:        PlayerColor (whose turn to move)
    play_ply:    int (default = inferred from pieces; only matters for
                 endgame eval weighting)
    best:        list of acceptable best actions; runner accepts any of them
    tag:         category label ("eat" / "cascade-off-edge" / "defense" / ...)

`best` actions are built with the same dataclasses the agent uses:
    PlaceAction(Coord(r, c))
    MoveAction(Coord(r, c), Direction.X)
    EatAction(Coord(r, c), Direction.X)
    CascadeAction(Coord(r, c), Direction.X)
"""

from referee.game import (
    Coord, Direction,
    EatAction, CascadeAction, MoveAction,
    PlayerColor,
)


U, D, L, R = Direction.Up, Direction.Down, Direction.Left, Direction.Right


PUZZLES = [
    # ---------- 1: trivial EAT -------------------------------------------------
    {
        "name": "eat-equal-height",
        "desc": "Red can eat an adjacent equal-height blue stack.",
        "pieces": {(3, 3): +2, (3, 4): -2,
                   (0, 0): +1, (7, 7): -1,
                   (5, 5): +1, (5, 6): -1},
        "turn": PlayerColor.RED,
        "best": [EatAction(Coord(3, 3), R)],
        "tag": "eat",
    },
    # ---------- 2: EAT preferred over MOVE ------------------------------------
    {
        "name": "eat-over-move",
        "desc": "Red has both an EAT and a passive MOVE; should choose EAT.",
        "pieces": {(2, 2): +3, (2, 3): -2,
                   (6, 6): +1, (7, 7): -1,
                   (0, 7): +1, (7, 0): -1},
        "turn": PlayerColor.RED,
        "best": [EatAction(Coord(2, 2), R)],
        "tag": "eat",
    },
    # ---------- 3: cascade pushes enemy off edge ------------------------------
    {
        "name": "cascade-push-off-right",
        "desc": "Red cascades right; blue at (4,7) gets pushed off the board.",
        "pieces": {(4, 4): +3, (4, 7): -2,
                   (0, 0): +1, (7, 7): -1,
                   (1, 1): +1, (6, 6): -1},
        "turn": PlayerColor.RED,
        "best": [CascadeAction(Coord(4, 4), R)],
        "tag": "cascade-off-edge",
    },
    # ---------- 4: win by elimination via EAT ---------------------------------
    {
        "name": "elimination-by-eat",
        "desc": "Blue has only one piece; red can eat it for instant win.",
        "pieces": {(3, 3): +2, (3, 4): -1,
                   (5, 5): +2, (0, 0): +1},
        "turn": PlayerColor.RED,
        "best": [EatAction(Coord(3, 3), R)],
        "tag": "win-elimination",
    },
    # ---------- 5: defensive — avoid being eaten ------------------------------
    {
        "name": "defense-flee-from-eat",
        "desc": "Blue threatens to EAT red's (3,3) stack next turn; "
                "red must vacate or capture first. The unique safe move is "
                "to EAT blue's (3,4) stack since it is the threat.",
        "pieces": {(3, 3): +2, (3, 4): -3,
                   (0, 0): +1, (7, 7): -1,
                   (6, 6): +1, (0, 7): -1},
        "turn": PlayerColor.RED,
        "best": [EatAction(Coord(3, 4), L),    # if blue acted first; symmetry
                 # Red's (3,3) height=2 can't actually eat blue's height-3 (3,4):
                 # so the unique safe action is to *move away*:
                 MoveAction(Coord(3, 3), U),
                 MoveAction(Coord(3, 3), D),
                 MoveAction(Coord(3, 3), L),
                 CascadeAction(Coord(3, 3), U),
                 CascadeAction(Coord(3, 3), D),
                 CascadeAction(Coord(3, 3), L)],
        "tag": "defense",
    },
    # ---------- 6: cascade hits enemy and pushes a chain ----------------------
    {
        "name": "cascade-chain-push",
        "desc": "Red cascades right; pushes a chain of blue stacks; the last "
                "gets pushed off the edge.",
        "pieces": {(4, 2): +4, (4, 4): -1, (4, 5): -1, (4, 7): -1,
                   (0, 0): +1, (7, 7): +1},
        "turn": PlayerColor.RED,
        "best": [CascadeAction(Coord(4, 2), R)],
        "tag": "cascade-off-edge",
    },
    # ---------- 7: EAT highest-value target (MVV-LVA) ------------------------
    {
        "name": "mvv-lva-pick-bigger",
        "desc": "Two EATs available; the bigger target should be preferred.",
        "pieces": {(3, 3): +4, (3, 4): -3, (3, 2): -1,
                   (0, 0): +1, (7, 7): -1},
        "turn": PlayerColor.RED,
        "best": [EatAction(Coord(3, 3), R)],
        "tag": "eat",
    },
    # ---------- 8: cascade better than MOVE (creates eat threat) -------------
    {
        "name": "cascade-create-threat",
        "desc": "A cascade spreads tokens to create multi-direction threats.",
        "pieces": {(2, 2): +3, (2, 4): -1, (4, 2): -1,
                   (0, 0): +1, (7, 7): -1, (6, 6): +1, (0, 7): -1},
        "turn": PlayerColor.RED,
        "best": [CascadeAction(Coord(2, 2), R),
                 CascadeAction(Coord(2, 2), D)],
        "tag": "cascade-threat",
    },
    # ---------- 9: don't self-eliminate ---------------------------------------
    {
        "name": "avoid-self-elimination",
        "desc": "Red has 2 pieces; cascading the bigger one off the edge would "
                "lose half its material. Should not cascade off the board.",
        "pieces": {(0, 4): +4, (5, 5): +1,
                   (7, 7): -2, (4, 4): -1},
        "turn": PlayerColor.RED,
        # Any action that's not "cascade up off the board" is acceptable.
        # We'll express this as a forbidden move; runner checks via 'forbid'.
        "best": [],
        "forbid": [CascadeAction(Coord(0, 4), U)],
        "tag": "blunder-avoidance",
    },
    # ---------- 10: mate-in-1 — final eat eliminates last blue piece ----------
    {
        "name": "mate-in-1",
        "desc": "Red's last legal EAT removes blue's last piece on the board.",
        "pieces": {(3, 3): +3, (4, 3): -2, (1, 0): +1, (5, 7): +1},
        "turn": PlayerColor.RED,
        "best": [EatAction(Coord(3, 3), D)],
        "tag": "win-elimination",
    },
]
