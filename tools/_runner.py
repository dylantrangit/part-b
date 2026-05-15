"""
Shared game-running primitives for the bench / SPRT / ratings / strength
tools. Subprocesses the referee, parses its result line, and returns a
structured outcome.

All tools in tools/ share this so wins/losses are counted identically
across analyses.
"""

import concurrent.futures
import re
import subprocess
import time
from dataclasses import dataclass

RESULT_RE = re.compile(r"result:\s*(player\s+\d|draw)\s*(?:\[([^\]]+)\])?", re.IGNORECASE)
PLAY_LINE_RE = re.compile(r"(RED|BLUE) plays action (\w+)\(([^)]*)\)")
TURN_LINE_RE = re.compile(r"(RED|BLUE) to play \(turn (\d+)\)")


@dataclass
class GameResult:
    red: str
    blue: str
    winner: str | None         # module name, or None
    draw: bool
    elapsed: float
    rc: int
    transcript: list[str]      # raw action strings, in order, like "RED PLACE 3-3"

    def outcome_for(self, module: str) -> float:
        """Win=1, draw=0.5, loss=0 from `module`'s perspective (must be red or blue)."""
        if self.draw:
            return 0.5
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == module else 0.0


def play_one(red: str, blue: str, time_limit: float, verbose: int = 1) -> GameResult:
    """Run a single referee subprocess and parse its output."""
    cmd = ["python", "-m", "referee", "-v", str(verbose), "-t", str(time_limit), red, blue]
    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - start
    output = (proc.stdout or "") + (proc.stderr or "")

    transcript: list[str] = []
    winner = None
    draw = False
    for line in output.splitlines():
        pm = PLAY_LINE_RE.search(line)
        if pm:
            transcript.append(f"{pm.group(1)} {pm.group(2)} {pm.group(3)}")
            continue
        rm = RESULT_RE.search(line)
        if rm:
            tag = rm.group(1).lower().strip()
            if "draw" in tag:
                draw = True
            else:
                bracket = rm.group(2) or ""
                if ":" in bracket:
                    winner = bracket.split(":", 1)[0]
                elif "1" in tag:
                    winner = red
                elif "2" in tag:
                    winner = blue

    return GameResult(
        red=red, blue=blue,
        winner=winner, draw=draw,
        elapsed=elapsed, rc=proc.returncode,
        transcript=transcript,
    )


def run_pairings(pairings, parallel: int, time_limit: float, verbose: int = 1,
                 on_result=None):
    """
    Run `pairings` (iterable of (red, blue) tuples) concurrently.
    Yields GameResults as they complete.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(play_one, r, b, time_limit, verbose) for r, b in pairings]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if on_result is not None:
                on_result(res)
            yield res
