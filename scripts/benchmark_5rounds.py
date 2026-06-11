"""
Benchmark 5 Matchups: 4 seeds x 4 corners = 16 matches per matchup.
Run in parallel using 4 workers.
"""

import sys
import time
import os
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trueskill
from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents, compute_ranks

ROUNDS = [
    {
        "name": "Round 1: codex/15 vs codex/13 vs Tactical vs Genius",
        "paths": [
            "agent/codex/15.py",
            "agent/codex/13.py",
            "TacticalRuleAgent",
            "GeniusRuleAgent",
        ],
    },
    {
        "name": "Round 2: codex/15 vs claude/2 vs Tactical vs Genius",
        "paths": [
            "agent/codex/15.py",
            "agent/claude/2.py",
            "TacticalRuleAgent",
            "GeniusRuleAgent",
        ],
    },
    {
        "name": "Round 3: codex/15 vs codex/13 vs claude/2 vs Tactical",
        "paths": [
            "agent/codex/15.py",
            "agent/codex/13.py",
            "agent/claude/2.py",
            "TacticalRuleAgent",
        ],
    },
    {
        "name": "Round 4: codex/15 vs codex/7 vs Tactical vs Genius",
        "paths": [
            "agent/codex/15.py",
            "agent/codex/7.py",
            "TacticalRuleAgent",
            "GeniusRuleAgent",
        ],
    },
    {
        "name": "Round 5: codex/15 vs codex/7 vs claude/2 vs Tactical",
        "paths": [
            "agent/codex/15.py",
            "agent/codex/7.py",
            "agent/claude/2.py",
            "TacticalRuleAgent",
        ],
    },
]

NUM_SEEDS = 4
NUM_CORNERS = 4
MAX_STEPS = 500
BASE_SEED = 42


def rotate(lst, n):
    """Rotate list right by n positions."""
    return lst[-n:] + lst[:-n]


def run_single_match(args):
    """
    Run a single match in parallel.
    args: (agent_paths, seat_order, seed, max_steps, match_idx)
    """
    agent_paths, seat_order, seed, max_steps, match_idx = args
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from engine.game import BomberEnv
    from scripts.participant.run_local_match import make_agents, compute_ranks

    ordered_paths = [agent_paths[seat_order[s]] for s in range(4)]
    agents, names = make_agents(ordered_paths, seed=seed)

    env = BomberEnv(max_steps=max_steps)
    obs = env.reset(seed=seed)
    step = 0
    death_steps = {}
    prev_alive = [bool(p[2]) for p in obs["players"]]

    while True:
        actions = []
        for seat in range(4):
            if int(obs["players"][seat][2]) == 1:
                try:
                    actions.append(agents[seat].act(obs))
                except Exception:
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

    return {
        'match_idx': match_idx,
        'ranks': ranks,
        'stats': stats_list,
        'names': names
    }


def get_abs_path(p):
    if p in ["TacticalRuleAgent", "GeniusRuleAgent", "SmarterRuleAgent", "SimpleRuleAgent", "RandomAgent"]:
        return p
    return str(Path(__file__).resolve().parents[1] / p)


def run_round(rnd):
    name = rnd["name"]
    raw_paths = rnd["paths"]
    agent_paths = [get_abs_path(p) for p in raw_paths]

    n = 4
    # TrueSkill (BTC defaults)
    ts_env = trueskill.TrueSkill(mu=100.0, sigma=33.333, draw_probability=0.1)
    ratings = [ts_env.Rating() for _ in range(n)]

    total_ranks = [0.0] * n
    wins = [0] * n
    draws = [0] * n
    scores = [0.0] * n
    total_kills = [0.0] * n
    total_boxes = [0.0] * n
    total_items = [0.0] * n
    total_bombs = [0.0] * n

    # Generate the 16 tasks for this round
    tasks = []
    match_idx = 0
    for corner in range(NUM_CORNERS):
        seat_order = rotate(list(range(n)), corner)
        for seed_idx in range(NUM_SEEDS):
            seed = BASE_SEED + corner * 10 + seed_idx * 7
            tasks.append((agent_paths, seat_order, seed, MAX_STEPS, match_idx))
            match_idx += 1

    total_matches = len(tasks)
    results = [None] * total_matches

    # Run tasks in parallel (max_workers=4)
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_match, task): task[-1] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results[res['match_idx']] = res
            completed += 1
            print(f"  [{completed}/{total_matches}] Match {res['match_idx']} completed: {res['names']}", flush=True)

    # Resolve agent names from the first match
    agent_labels = []
    first_names = results[0]['names']
    for label in first_names:
        # Simplify folder names
        if "/" in label or "\\" in label:
            p = Path(label)
            agent_labels.append(f"{p.parent.name}/{p.stem}" if p.parent.name in ["codex", "claude"] else p.stem)
        else:
            agent_labels.append(label)

    # Replay results sequentially
    match_idx = 0
    for corner in range(NUM_CORNERS):
        seat_order = rotate(list(range(n)), corner)
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

    # Compute final scores
    for i in range(n):
        scores[i] = ratings[i].mu - 3 * ratings[i].sigma

    print(f"\n==============================================================================")
    print(f"  {name}")
    print(f"==============================================================================")
    header = f"  {'Rank':>4} {'Agent':<25} {'Score':>8} {'Wins':>6} {'Draws':>6} {'AvgRank':>8} {'AvgKill':>8} {'AvgBox':>7} {'AvgItem':>8} {'AvgBomb':>8}"
    print(header)
    print("  " + "-" * 76)
    
    order = sorted(range(n), key=lambda i: -scores[i])
    for rank_pos, idx in enumerate(order):
        label = agent_labels[idx]
        avg_rank = total_ranks[idx] / total_matches
        avg_kill = total_kills[idx] / total_matches
        avg_box = total_boxes[idx] / total_matches
        avg_item = total_items[idx] / total_matches
        avg_bomb = total_bombs[idx] / total_matches
        medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(rank_pos, "   ")
        print(f"  {medal} {rank_pos+1:<2} {label:<25} {scores[idx]:>8.2f} {wins[idx]:>6} {draws[idx]:>6} {avg_rank:>8.3f}  {avg_kill:>7.2f} {avg_box:>6.1f} {avg_item:>7.2f} {avg_bomb:>7.2f}")
    print("  " + "-" * 76)


def main():
    total_start = time.time()
    for idx, rnd in enumerate(ROUNDS):
        print(f"\n>>> Starting Round {idx+1}/{len(ROUNDS)}: {rnd['name']}...", flush=True)
        run_round(rnd)
    total_elapsed = time.time() - total_start
    print(f"\nTotal benchmark complete: {len(ROUNDS) * NUM_SEEDS * NUM_CORNERS} matches in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
