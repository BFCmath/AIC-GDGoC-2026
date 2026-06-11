"""
Benchmark: codex v14 (MetaFarmSwitchV1) vs codex v7 vs codex v13 vs claude v2
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.game import BomberEnv
from competition.evaluation.runtime_guard import load_agent_instance

AGENTS = [
    ("codex v14", "agent/codex/14/agent.py"),
    ("codex v7",   "agent/codex/7.py"),
    ("codex v13",  "agent/codex/13.py"),
    ("claude v2",  "agent/claude/2.py"),
]

NUM_EPISODES = 50
MAX_STEPS = 500
BASE_SEED = 42


def compute_ranks(survivors, death_order, env):
    ranks = [None] * 4
    if len(survivors) == 0:
        r = 0
        for j in reversed(death_order):
            ranks[j] = r; r += 1
        return ranks
    if len(survivors) == 1:
        ranks[survivors[0]] = 0
    else:
        def tb_key(i):
            s = env.players[i].stats
            return (s['kills'], s['boxes'], s['items'], s['bombs'])
        sorted_surv = sorted(survivors, key=tb_key, reverse=True)
        rv = 0
        for idx, pi in enumerate(sorted_surv):
            if idx > 0 and tb_key(pi) < tb_key(sorted_surv[idx-1]):
                rv = idx
            ranks[pi] = rv
    nsr = max(ranks[i] for i in survivors) + 1
    cr = nsr
    for j in reversed(death_order):
        ranks[j] = cr; cr += 1
    return ranks


def main():
    agents = []
    team_ids = []
    for label, path in AGENTS:
        full_path = str(Path(__file__).resolve().parents[1] / path)
        agent = load_agent_instance(full_path, len(agents))
        agents.append(agent)
        team_ids.append(agent.team_id)
        print(f"  {label:12s} -> {agent.team_id}")

    # Stats accumulators
    total_ranks = [0] * 4
    wins = [0] * 4
    draws = [0] * 4
    total_kills = [0] * 4
    total_boxes = [0] * 4
    total_items = [0] * 4
    total_bombs = [0] * 4
    survival_steps_sum = [0] * 4
    episode_times = []

    print(f"\nRunning {NUM_EPISODES} episodes...")

    for ep in range(NUM_EPISODES):
        seed = BASE_SEED + ep * 7
        env = BomberEnv(max_steps=MAX_STEPS)
        obs = env.reset(seed=seed)
        death_order = []
        prev_alive = [bool(p[2]) for p in obs["players"]]
        start = time.time()

        while True:
            done = False
            actions = []
            for i in range(4):
                if int(obs["players"][i][2]) == 1:
                    try:
                        actions.append(agents[i].act(obs))
                    except Exception as e:
                        print(f"  Agent {i} error ep {ep}: {e}")
                        actions.append(0)
                else:
                    actions.append(0)
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated

            alive_now = [bool(p[2]) for p in obs["players"]]
            for i in range(4):
                if prev_alive[i] and not alive_now[i]:
                    death_order.append(i)
            prev_alive = alive_now

            if done:
                break

        elapsed = time.time() - start
        episode_times.append(elapsed)

        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [i for i in range(4) if alive_final[i]]
        ranks = compute_ranks(survivors, death_order, env)

        winners = [i for i in range(4) if ranks[i] == 0]
        for i in range(4):
            total_ranks[i] += ranks[i]
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
            survival_steps_sum[i] += (env.current_step if not alive_final[i] else MAX_STEPS)

        winner_names = [AGENTS[w][0] for w in winners]
        if len(winners) == 1:
            print(f"  Ep {ep+1:2d}: {winner_names[0]} wins (seed={seed})")
        else:
            print(f"  Ep {ep+1:2d}: Draw {winner_names} (seed={seed})")

    # Report
    n = NUM_EPISODES
    avg_time = sum(episode_times) / n
    print(f"\n{'='*90}")
    print(f"{'Agent':<14} {'Win%':>7} {'Draw%':>7} {'AvgRank':>8} {'AvgKill':>8} {'AvgBox':>7} {'AvgItem':>8} {'AvgBomb':>8} {'AvgSurv':>8}")
    print(f"{'-'*90}")
    for i, (label, _) in enumerate(AGENTS):
        print(f"{label:<14} {wins[i]/n*100:>6.1f}% {draws[i]/n*100:>6.1f}% {total_ranks[i]/n:>8.3f}  {total_kills[i]/n:>7.2f} {total_boxes[i]/n:>6.1f} {total_items[i]/n:>7.2f} {total_bombs[i]/n:>7.2f} {survival_steps_sum[i]/n:>7.0f}")
    print(f"{'-'*90}")
    print(f"Avg episode time: {avg_time:.2f}s | Total: {n} episodes | Max steps: {MAX_STEPS}")


if __name__ == "__main__":
    main()
