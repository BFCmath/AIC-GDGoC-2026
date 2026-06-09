"""
benchmark.py — Head-to-head benchmark for multiple agents.

Implements BTC-accurate ranking:
  - Tie-break (kills > boxes > items > bombs) when ≥2 agents survive at step limit.
  - 100ms/step timeout enforcement matching the BTC server constraint.

Usage:
    # Full benchmark with timeout (default)
    python -m scripts.participant.benchmark \\
        --agents agent/claude/1.py agent/codex/1.py TacticalRuleAgent \\
        --matches 30 --max_steps 400 --no-limit

    # Disable timeout (for debugging)
    python -m scripts.participant.benchmark \\
        --agents agent/claude/1.py GeniusRuleAgent \\
        --matches 10 --no-timeout

    # Force truncated games to test tie-break
    python -m scripts.participant.benchmark \\
        --agents agent/claude/1.py GeniusRuleAgent TacticalRuleAgent SmarterRuleAgent \\
        --matches 10 --max_steps 10 --no-limit
"""

import os
import sys
import argparse
import concurrent.futures
import random
import time
from pathlib import Path

# Simulate single-threaded production constraints by default
_no_limit = "--no-limit" in sys.argv
if not _no_limit:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import trueskill
from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents, compute_ranks

# ── ANSI colour helpers ───────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"
ORANGE = "\033[33m"

PLAYER_COLORS = ["\033[91m", "\033[94m", "\033[92m", "\033[93m"]

def colorize(text, color):
    return f"{color}{text}{RESET}"


# ── Timeout wrapper ───────────────────────────────────────────────────────────

def timed_act(executor, agent, obs, timeout_ms: int = 100):
    """
    Call agent.act(obs) with a wall-clock timeout.

    Returns (action, elapsed_ms, timed_out).
    On timeout → action=0 (STOP), timed_out=True.

    Note: Python cannot forcibly kill the running thread. The agent's act()
    continues in background but its result is discarded. A persistent executor
    (max_workers=1) ensures next calls queue correctly.
    """
    t0 = time.perf_counter()
    future = executor.submit(agent.act, obs)
    try:
        action = future.result(timeout=timeout_ms / 1000.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return action, elapsed_ms, False
    except concurrent.futures.TimeoutError:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return 0, elapsed_ms, True


# ── Core benchmark logic ──────────────────────────────────────────────────────

def run_benchmark(
    agent_specs: list,
    num_matches: int = 50,
    max_steps: int = 400,
    seed: int | None = None,
    use_timeout: bool = True,
    timeout_ms: int = 100,
):
    n = len(agent_specs)
    assert 2 <= n <= 4, "Need 2–4 agents."

    # Pad to 4 players if fewer agents specified
    padded_specs = agent_specs + ["TacticalRuleAgent"] * (4 - n)

    env = BomberEnv(max_steps=max_steps, seed=seed)

    # TrueSkill (BTC defaults)
    ts_env = trueskill.TrueSkill(mu=100.0, sigma=33.333, draw_probability=0.1)
    ratings = [ts_env.Rating() for _ in range(n)]

    match_stats = [{"wins": 0, "draws": 0, "rank_sum": 0} for _ in range(n)]
    timeout_stats = [{"count": 0, "max_ms": 0.0} for _ in range(n)]
    active_steps = [0] * n   # steps where agent was alive and timed

    names = None

    # Header
    print()
    print(colorize("═" * 66, CYAN))
    timeout_note = f"  ⏱ timeout={timeout_ms}ms" if use_timeout else "  (timeout OFF)"
    print(colorize(
        f"  🎮  BOMBERLAND BENCHMARK  —  {num_matches} matches{timeout_note}",
        BOLD + CYAN,
    ))
    print(colorize("═" * 66, CYAN))

    # Persistent per-agent executors for timeout (avoids re-spawning threads per step)
    executors = {}
    if use_timeout:
        for i in range(n):
            executors[i] = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    t0 = time.time()

    try:
        for match_idx in range(num_matches):
            match_seed = None if seed is None else seed + match_idx
            try:
                agents, cur_names = make_agents(padded_specs, seed=match_seed)
            except Exception as e:
                print(colorize(f"\n[ERROR] Failed to load agents: {e}", RED))
                return

            if names is None:
                names = cur_names[:n]

            obs = env.reset(seed=match_seed)
            done = False
            step = 0
            prev_alive = [bool(p[2]) for p in obs["players"]]
            death_order = []   # indices in order of death

            while not done and step < max_steps:
                actions = []

                for j in range(4):
                    agent_alive = bool(obs["players"][j][2])

                    if not agent_alive:
                        actions.append(0)
                        continue

                    if use_timeout and j < n:
                        action, elapsed_ms, timed_out = timed_act(
                            executors[j], agents[j], obs, timeout_ms
                        )
                        active_steps[j] += 1
                        if timed_out:
                            timeout_stats[j]["count"] += 1
                            if elapsed_ms > timeout_stats[j]["max_ms"]:
                                timeout_stats[j]["max_ms"] = elapsed_ms
                    else:
                        try:
                            action = agents[j].act(obs)
                        except Exception:
                            action = 0

                    actions.append(action)

                obs, terminated, truncated = env.step(actions)
                done = terminated or truncated
                step += 1

                alive_now = [bool(p[2]) for p in obs["players"]]
                for j in range(4):
                    if prev_alive[j] and not alive_now[j]:
                        death_order.append(j)
                prev_alive = alive_now

            # ── Ranking ──────────────────────────────────────────────────────
            alive_final = [bool(p[2]) for p in obs["players"]]
            survivors = [j for j in range(4) if alive_final[j]]

            # BTC-accurate ranks (with tie-break applied when truncated + ≥2 survivors)
            ranks = compute_ranks(survivors, death_order, env)
            tb_applied = truncated and len(survivors) > 1

            # Win/Draw determination for tracked agents
            tracked_winners = [i for i in range(n) if ranks[i] == 0]

            for i in range(n):
                match_stats[i]["rank_sum"] += ranks[i]
                if ranks[i] == 0:
                    if len(tracked_winners) == 1:
                        match_stats[i]["wins"] += 1
                    else:
                        match_stats[i]["draws"] += 1

            # TrueSkill update (pad agents use throwaway ratings)
            ts_groups = [
                (ratings[j],) if j < n else (ts_env.Rating(),)
                for j in range(4)
            ]
            new_ratings = ts_env.rate(ts_groups, ranks=ranks)
            for i in range(n):
                ratings[i] = new_ratings[i][0]

            # ── Result label ─────────────────────────────────────────────────
            if len(tracked_winners) == 1:
                result_label = colorize(f"WIN → {names[tracked_winners[0]]}", GREEN)
            elif len(tracked_winners) > 1:
                result_label = colorize(
                    f"DRAW ({', '.join(names[w] for w in tracked_winners)})", YELLOW
                )
            else:
                result_label = colorize("ALL DEAD", DIM)

            if tb_applied:
                result_label += colorize(" [TB]", ORANGE)

            rank_strs = "  ".join(
                f"{PLAYER_COLORS[i]}{names[i][:12]}={ranks[i]}{RESET}"
                for i in range(n)
            )
            print(
                f"  Match {match_idx + 1:>3}/{num_matches} | {rank_strs}  | {result_label}",
                flush=True,
            )

    finally:
        for ex in executors.values():
            ex.shutdown(wait=False)

    elapsed = time.time() - t0

    # ── Final leaderboard ─────────────────────────────────────────────────────
    print()
    print(colorize("═" * 66, CYAN))
    print(colorize("  📊  FINAL LEADERBOARD", BOLD + CYAN))
    print(colorize("═" * 66, CYAN))

    col = [20, 6, 6, 9, 12]
    header = (
        f"  {'Agent':<{col[0]}} {'Wins':>{col[1]}} {'Draws':>{col[2]}} "
        f"{'AvgRank':>{col[3]}} {'TrueSkill':>{col[4]}}"
    )
    print(colorize(header, BOLD))
    print(colorize("  " + "─" * 64, DIM))

    leaderboard = sorted(
        range(n),
        key=lambda i: -(ratings[i].mu - 3 * ratings[i].sigma),
    )

    medals = ["🥇", "🥈", "🥉", "  "]
    colors = [GREEN, YELLOW, "\033[96m", DIM]

    for rank_pos, i in enumerate(leaderboard):
        score = ratings[i].mu - 3 * ratings[i].sigma
        avg_rank = match_stats[i]["rank_sum"] / num_matches
        c = colors[min(rank_pos, 3)]
        row = (
            f"  {medals[min(rank_pos, 3)]} {names[i]:<{col[0]-2}} "
            f"{match_stats[i]['wins']:>{col[1]}} "
            f"{match_stats[i]['draws']:>{col[2]}} "
            f"{avg_rank:>{col[3]}.2f} "
            f"{score:>{col[4]}.2f}"
        )
        print(colorize(row, c))

    print(colorize("  " + "─" * 64, DIM))
    print(
        f"\n  Matches: {num_matches}  |  Max steps/match: {max_steps}  |  "
        f"Elapsed: {elapsed:.1f}s"
    )

    # ── Timeout summary ───────────────────────────────────────────────────────
    if use_timeout:
        print()
        print(colorize("  ⏱  TIMEOUT SUMMARY  (BTC server limit: 100ms/step)", BOLD + CYAN))
        print(colorize("  " + "─" * 64, DIM))

        any_violation = False
        for i in range(n):
            ts = timeout_stats[i]
            total = active_steps[i]
            pct = (ts["count"] / total * 100) if total > 0 else 0.0
            if ts["count"] > 0:
                status = colorize("⚠ VIOLATION", RED + BOLD)
                any_violation = True
            else:
                status = colorize("✓  OK      ", GREEN)
            print(
                f"  {names[i]:<22} {status}  "
                f"timeouts={ts['count']}/{total} ({pct:.1f}%)  "
                f"max_spike={ts['max_ms']:.0f}ms"
            )

        if any_violation:
            print(colorize(
                "\n  ⚠  Agent(s) marked above will likely be penalized or disqualified on the BTC server!",
                RED,
            ))
        else:
            print(colorize("\n  All tracked agents are within the 100ms/step limit. ✓", GREEN))
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Head-to-head benchmark for 2–4 agents. "
            "Implements BTC-accurate tie-break ranking and 100ms/step timeout enforcement."
        )
    )
    parser.add_argument(
        "--agents", nargs="+", required=True,
        help=(
            "Agent paths or baseline names "
            "(e.g. agent/claude/1.py GeniusRuleAgent). 2–4 agents."
        ),
    )
    parser.add_argument("--matches", type=int, default=50, help="Number of matches (default: 50).")
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps per match (default: 400).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--no-limit", action="store_true", help="Disable single-thread restrictions.")

    timeout_grp = parser.add_mutually_exclusive_group()
    timeout_grp.add_argument(
        "--timeout", dest="use_timeout", action="store_true", default=True,
        help="Enforce 100ms/step timeout (default: ON).",
    )
    timeout_grp.add_argument(
        "--no-timeout", dest="use_timeout", action="store_false",
        help="Disable timeout enforcement (faster, less accurate).",
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=100,
        help="Timeout threshold in ms (default: 100, matching BTC server).",
    )

    args = parser.parse_args()

    run_benchmark(
        agent_specs=args.agents,
        num_matches=args.matches,
        max_steps=args.max_steps,
        seed=args.seed,
        use_timeout=args.use_timeout,
        timeout_ms=args.timeout_ms,
    )
