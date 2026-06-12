"""
Benchmark 6 agents: rotate through 4-player groups, corners, and seeds.

  - 3 complementary 4-agent groups covering all 6 agents
  - 4 corner rotations per group
  - 4 random seeds
  - Total 3 * 4 * 4 = 48 matches (each agent plays 32 matches)
  - TrueSkill scoring (BTC defaults)
  - Report final table sorted by Score (mu - 3*sigma)
"""

import sys
import time
import os
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trueskill
from engine.game import BomberEnv
from competition.evaluation.runtime_guard import load_agent_instance

AGENT_SPECS = [
    "agent/fable/1.py",
    "agent/codex/15.py",
    "agent/claude/2.py",
    "agent/codex/7.py",
    "agent/smarter_rule_agent.py",
    "agent/genius_rule_agent.py",
]

import itertools
# All 15 combinations of 4 agents from the 6-agent pool (6C4 = 15)
GROUPS = [list(g) for g in itertools.combinations(range(6), 4)]

NUM_SEEDS = 4
NUM_CORNERS = 4
MAX_STEPS = 500
BASE_SEED = 42


def rotate(lst, n):
    """Rotate list right by n positions."""
    return lst[-n:] + lst[:-n]


def compute_ranks(survivors, death_steps, env):
    ranks = [0] * 4
    if survivors:
        def tb_key(i):
            s = env.players[i].stats
            return (s['kills'], s['boxes'], s['items'], s['bombs'])
        ordered = sorted(survivors, key=tb_key, reverse=True)
        rv = 0
        for idx, pi in enumerate(ordered):
            if idx > 0 and tb_key(pi) < tb_key(ordered[idx - 1]):
                rv = idx
            ranks[pi] = rv
    dead = [i for i in range(4) if i not in survivors]
    if dead:
        base = max((ranks[i] for i in survivors), default=-1) + 1
        dead_sorted = sorted(dead, key=lambda i: death_steps.get(i, 0), reverse=True)
        cr = base
        for idx, pid in enumerate(dead_sorted):
            if idx > 0 and death_steps.get(pid, 0) < death_steps.get(dead_sorted[idx - 1], 0):
                cr = base + idx
            ranks[pid] = cr
    return ranks


def run_single_match(args):
    """
    Run a single match in a parallel subprocess.
    args is a tuple: (abs_paths, seat_order, seed, max_steps, match_idx)
    """
    abs_paths, seat_order, seed, max_steps, match_idx = args
    import os
    import time
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from engine.game import BomberEnv
    from competition.evaluation.runtime_guard import load_agent_instance

    t_start = time.time()
    # Load agent instances with correct seat indices (agent_id)
    agents = []
    for seat in range(4):
        agent_idx = seat_order[seat]
        agent_path = abs_paths[agent_idx]
        agents.append(load_agent_instance(agent_path, seat))

    env = BomberEnv(max_steps=max_steps)
    obs = env.reset(seed=seed)
    step = 0
    death_steps = {}
    prev_alive = [bool(p[2]) for p in obs["players"]]

    agent_names = []
    for s in range(4):
        p_parts = Path(abs_paths[seat_order[s]]).parts
        if p_parts[-1] == 'agent.py':
            name = f"{p_parts[-3]}/{p_parts[-2]}"
        else:
            name = f"{p_parts[-2]}/{p_parts[-1].replace('.py', '')}"
        agent_names.append(name)

    while True:
        actions = []
        for seat in range(4):
            if int(obs["players"][seat][2]) == 1:
                try:
                    actions.append(agents[seat].act(obs))
                except Exception as e:
                    actions.append(0)
            else:
                actions.append(0)
        obs, terminated, truncated = env.step(actions)
        step += 1
        alive_now = [bool(p[2]) for p in obs["players"]]
        for seat in range(4):
            if prev_alive[seat] and not alive_now[seat]:
                death_steps[seat] = step
        prev_alive = alive_now
        if terminated or truncated:
            break

    alive_final = [bool(p[2]) for p in obs["players"]]
    survivors = [s for s in range(4) if alive_final[s]]
    ranks = compute_ranks(survivors, death_steps, env)

    stats_list = []
    for seat in range(4):
        stats = env.players[seat].stats
        stats_list.append({
            'kills': stats['kills'],
            'boxes': stats['boxes'],
            'items': stats['items'],
            'bombs': stats['bombs']
        })

    t_elapsed = time.time() - t_start
    return {
        'match_idx': match_idx,
        'ranks': ranks,
        'stats': stats_list,
        'names': agent_names,
        'steps': step,
        'elapsed': t_elapsed
    }


def run_benchmark():
    n_total = len(AGENT_SPECS)
    labels = []
    abs_paths = []
    for spec in AGENT_SPECS:
        p = Path(__file__).resolve().parents[1] / spec
        if p.is_dir():
            p = p / "agent.py"
        abs_paths.append(str(p))
        parent = Path(spec).parent.name
        stem = Path(spec).stem
        label = f"{parent}/{stem}" if parent in ["codex", "claude", "fable"] else stem
        labels.append(label)

    ts_env = trueskill.TrueSkill(mu=100.0, sigma=33.333, draw_probability=0.1)
    ratings = [ts_env.Rating() for _ in range(n_total)]

    total_ranks = [0.0] * n_total
    wins = [0] * n_total
    draws = [0] * n_total
    match_count = [0] * n_total
    total_kills = [0.0] * n_total
    total_boxes = [0.0] * n_total
    total_items = [0.0] * n_total
    total_bombs = [0.0] * n_total

    t0 = time.time()

    # Generate all match tasks
    tasks = []
    match_idx = 0
    for gid, group in enumerate(GROUPS):
        for corner in range(NUM_CORNERS):
            seat_order = rotate(group, corner)
            for seed_idx in range(NUM_SEEDS):
                seed = BASE_SEED + gid * 100 + corner * 10 + seed_idx * 7
                tasks.append((abs_paths, seat_order, seed, MAX_STEPS, match_idx))
                match_idx += 1

    total_matches = len(tasks)

    print("=" * 78)
    print(f"  6-Agent Pool Rotate Benchmark  —  {total_matches} matches in parallel (max_workers=4)")
    print(f"  Agents: {', '.join(labels)}")
    print("=" * 78)

    results = [None] * total_matches

    # Run matches in parallel
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_match, task): task[-1] for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results[res['match_idx']] = res
            completed += 1
            print(f"  [{completed}/{total_matches}] Match {res['match_idx']+1} completed: {res['names']} in {res['elapsed']:.1f}s (steps={res['steps']})", flush=True)

    # Replay results sequentially to update TrueSkill and statistics in correct order
    match_idx = 0
    for gid, group in enumerate(GROUPS):
        for corner in range(NUM_CORNERS):
            seat_order = rotate(group, corner)
            for seed_idx in range(NUM_SEEDS):
                res = results[match_idx]
                ranks = res['ranks']
                stats_list = res['stats']

                # TrueSkill update (map seat rank → global agent index)
                seat_order_for_ts = [(ratings[seat_order[s]],) for s in range(4)]
                new_ratings = ts_env.rate(seat_order_for_ts, ranks=ranks)
                for s in range(4):
                    ratings[seat_order[s]] = new_ratings[s][0]

                winners = [seat_order[s] for s in range(4) if ranks[s] == 0]

                for s in range(4):
                    agent_idx = seat_order[s]
                    total_ranks[agent_idx] += ranks[s]
                    match_count[agent_idx] += 1
                    if ranks[s] == 0:
                        if len(winners) == 1:
                            wins[agent_idx] += 1
                        else:
                            draws[agent_idx] += 1
                    total_kills[agent_idx] += stats_list[s]['kills']
                    total_boxes[agent_idx] += stats_list[s]['boxes']
                    total_items[agent_idx] += stats_list[s]['items']
                    total_bombs[agent_idx] += stats_list[s]['bombs']

                match_idx += 1

    elapsed = time.time() - t0

    # ── Final report ──
    scores = [ratings[i].mu - 3 * ratings[i].sigma for i in range(n_total)]
    order = sorted(range(n_total), key=lambda i: -scores[i])

    print("\n" + "=" * 78)
    print(f"  FINAL RANKING  —  {total_matches} matches in {elapsed:.1f}s")
    print("=" * 78)
    header = f"  {'Rank':>4} {'Agent':<22} {'Score':>8} {'Wins':>6} {'Draws':>6} {'AvgRank':>8} {'AvgKill':>8} {'AvgBox':>7} {'AvgItem':>8} {'AvgBomb':>8}"
    print(header)
    print("  " + "-" * 78)
    for rank_pos, idx in enumerate(order):
        label = labels[idx]
        mc = match_count[idx]
        avg_rank = total_ranks[idx] / mc if mc else 0
        avg_kill = total_kills[idx] / mc if mc else 0
        avg_box = total_boxes[idx] / mc if mc else 0
        avg_item = total_items[idx] / mc if mc else 0
        avg_bomb = total_bombs[idx] / mc if mc else 0
        medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(rank_pos, "   ")
        print(f"  {medal} {rank_pos+1:<2} {label:<22} {scores[idx]:>8.2f} {wins[idx]:>6} {draws[idx]:>6} {avg_rank:>8.3f}  {avg_kill:>7.2f} {avg_box:>6.1f} {avg_item:>7.2f} {avg_bomb:>7.2f}")
    print("  " + "-" * 78)
    print(f"\n  Matches per agent: {[match_count[i] for i in range(n_total)]}")
    print()


if __name__ == "__main__":
    run_benchmark()
