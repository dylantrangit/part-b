"""Profile mcts_final from a representative mid-game state.

Reports top hot functions by cumulative time, iters/s, and counts for the
new I6 pieces (hybrid calls, root trim, etc.) so we can attribute the
zero-uplift bench result.
"""
import argparse
import cProfile
import io
import pstats
import random
import time

from agent.core.board import GameState
from agent.core.placement import choose_placement_action
from agent.core.time_budget import TimeBudget
from agent.search import mcts_final as mf
from agent.search import mcts_heavy as mh


def warm_up_state(seed: int = 42, play_plies: int = 20):
    random.seed(seed)
    state = GameState()
    for _ in range(8):
        state.apply(choose_placement_action(state))

    root = None
    for _ in range(play_plies):
        if not list(state.legal_actions()):
            break
        action, root = mf.mcts_final(state, TimeBudget(0.3), root=root)
        if action is None:
            break
        state.apply(action)
        if root is not None and action in root.children:
            child = root.children[action]
            child.parent = None
            root = child
        else:
            root = None
    return state, root


def profile_one(backend: str, state: GameState, root, budget_s: float = 5.0):
    profiler = cProfile.Profile()
    start = time.monotonic()
    profiler.enable()
    if backend == "final":
        action, new_root = mf.mcts_final(state, TimeBudget(budget_s), root=root)
        total_visits = sum(c.visits for c in new_root.children.values())
        n_children = len(new_root.children)
    else:
        action = mh.mcts(state.copy(), TimeBudget(budget_s))
        total_visits = -1  # mcts_heavy doesn't expose its tree
        n_children = -1
    profiler.disable()
    elapsed = time.monotonic() - start

    print(f"\n--- {backend} ---")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Root-child visits: {total_visits}")
    if total_visits > 0:
        print(f"Iter/s: {total_visits / elapsed:.0f}")
    print(f"Expanded children: {n_children}")
    print(f"Chosen: {action}")

    s = io.StringIO()
    pstats.Stats(profiler, stream=s).strip_dirs().sort_stats("cumulative").print_stats(20)
    print(s.getvalue())

    s2 = io.StringIO()
    pstats.Stats(profiler, stream=s2).strip_dirs().print_stats(
        "_hybrid_value|_pvs|_trim_to_cap|_filter_root_safe|heavy_rollout"
        "|_selection_score|_materialise|expand|_try_prove|_select_child"
    )
    print(s2.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["final", "heavy", "both"], default="both")
    parser.add_argument("--budget", type=float, default=5.0)
    args = parser.parse_args()

    state, root = warm_up_state()
    print(
        f"Warmed state: play_ply={state.play_ply}, "
        f"red_tokens={state.red_tokens}, blue_tokens={state.blue_tokens}, "
        f"reuse root: {'yes' if root else 'no'}"
        + (f" ({len(root.children)} children)" if root else "")
    )

    if args.backend in ("final", "both"):
        profile_one("final", state, root, args.budget)
    if args.backend in ("heavy", "both"):
        profile_one("heavy", state, None, args.budget)


if __name__ == "__main__":
    main()
