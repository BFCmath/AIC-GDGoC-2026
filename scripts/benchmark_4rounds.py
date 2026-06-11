"""
Benchmark: 4 rounds × 10 eps = 40 eps total

Round 1: codex15 vs codex7 vs codex13 vs claude2
Round 2: codex15 + codex7 vs Smarter vs Tactical
Round 3: codex15 + codex13 vs Smarter vs Tactical
Round 4: codex15 + codex2 vs Smarter vs Tactical
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trueskill
from engine.game import BomberEnv
from competition.evaluation.runtime_guard import load_agent_instance

ROUNDS = [
    {
        "name": "R1: codex15 vs v7 vs v13 vs claude2",
        "paths": [
            "agent/codex/15.py",
            "agent/codex/7.py",
            "agent/codex/13.py",
            "agent/claude/2.py",
        ],
    },
    {
        "name": "R2: codex15+codex7 vs Smarter vs Tactical",
        "paths": [
            "agent/codex/15.py",
            "agent/codex/7.py",
            "agent/smarter_rule_agent.py",
            "agent/tactical_rule_agent.py",
        ],
    },
    {
        "name": "R3: codex15+codex13 vs Smarter vs Tactical",
        "paths": [
            "agent/codex/15.py",
            "agent/codex/13.py",
            "agent/smarter_rule_agent.py",
            "agent/tactical_rule_agent.py",
        ],
    },
    {
        "name": "R4: codex15+codex2 vs Smarter vs Tactical",
        "paths": [
            "agent/codex/15.py",
            "agent/codex/2.py",
            "agent/smarter_rule_agent.py",
            "agent/tactical_rule_agent.py",
        ],
    },
]

NUM_EPISODES = 10
MAX_STEPS = 500
BASE_SEED = 42

def compute_ranks(survivors, death_steps, env):
    """BTC-accurate ranking with same-step death grouping."""
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


def run_round(label, agent_paths, num_eps):
    agents = []
    agent_labels = []
    for path in agent_paths:
        full = str(Path(__file__).resolve().parents[1] / path)
        agent = load_agent_instance(full, len(agents))
        agents.append(agent)
        agent_labels.append(f"{Path(path).parent.name}/{Path(path).stem}" if Path(path).parent.name != Path(path).stem else Path(path).stem)

    n = len(agents)

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

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    for ep in range(num_eps):
        seed = BASE_SEED + ep * 7
        env = BomberEnv(max_steps=MAX_STEPS)
        obs = env.reset(seed=seed)
        step = 0
        death_steps = {}
        prev_alive = [bool(p[2]) for p in obs["players"]]

        while True:
            actions = []
            for i in range(n):
                if int(obs["players"][i][2]) == 1:
                    try:
                        actions.append(agents[i].act(obs))
                    except Exception as e:
                        print(f"  Agent {i} error ep {ep}: {e}")
                        actions.append(0)
                else:
                    actions.append(0)
            obs, terminated, truncated = env.step(actions)
            step += 1
            alive_now = [bool(p[2]) for p in obs["players"]]
            for i in range(n):
                if prev_alive[i] and not alive_now[i]:
                    death_steps[i] = step
            prev_alive = alive_now
            if terminated or truncated:
                break

        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [i for i in range(n) if alive_final[i]]
        ranks = compute_ranks(survivors, death_steps, env)
        winners = [i for i in range(n) if ranks[i] == 0]

        # TrueSkill update
        ts_groups = [(ratings[i],) for i in range(n)]
        new_ratings = ts_env.rate(ts_groups, ranks=ranks)
        for i in range(n):
            ratings[i] = new_ratings[i][0]

        for i in range(n):
            total_ranks[i] += ranks[i]
            scores[i] = ratings[i].mu - 3 * ratings[i].sigma
            if ranks[i] == 0:
                if len(winners) == 1:
                    wins[i] += 1
                else:
                    draws[i] += 1
            s = env.players[i].stats
            total_kills[i] += s['kills']
            total_boxes[i] += s['boxes']
            total_items[i] += s['items']
            total_bombs[i] += s['bombs']

        winner_labels = [agent_labels[w] for w in winners]
        if len(winners) == 1:
            print(f"  Ep {ep+1:2d}: {winner_labels[0]} wins (seed={seed})")
        else:
            print(f"  Ep {ep+1:2d}: Draw {winner_labels} (seed={seed})")

    # Report (score-sorted)
    order = sorted(range(n), key=lambda i: -scores[i])
    print(f"\n{'Agent':<30} {'Score':>8} {'Win%':>7} {'Draw%':>7} {'AvgRank':>8} {'AvgKill':>8} {'AvgBox':>7} {'AvgItem':>8} {'AvgBomb':>8}")
    print(f"{'-'*86}")
    for idx, i in enumerate(order):
        label = agent_labels[i]
        medals = {0: "*", 1: " ", 2: " "}
        mark = medals.get(idx, " ")
        print(f"{mark}{label:<29} {scores[i]:>8.2f} {wins[i]/num_eps*100:>6.1f}% {draws[i]/num_eps*100:>6.1f}% {total_ranks[i]/num_eps:>8.3f}  {total_kills[i]/num_eps:>7.2f} {total_boxes[i]/num_eps:>6.1f} {total_items[i]/num_eps:>7.2f} {total_bombs[i]/num_eps:>7.2f}")
    print(f"{'='*70}\n")


def main():
    total_start = time.time()
    for rnd in ROUNDS:
        run_round(rnd["name"], rnd["paths"], NUM_EPISODES)
    total_elapsed = time.time() - total_start
    print(f"\nTotal: {len(ROUNDS) * NUM_EPISODES} episodes in {total_elapsed:.1f}s")

if __name__ == "__main__":
    main()
