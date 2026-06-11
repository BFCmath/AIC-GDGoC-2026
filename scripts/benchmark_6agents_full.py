"""
Full round-robin benchmark: 6 agents (v2, v4, v7, v8, v13, v15)
All C(6,4)=15 combos × 4 seeds × 4 corner rotations = 240 matches.
Parallel workers (RAM-capped at 4).
"""
import sys, time, itertools, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

AGENTS = [
    ("codex7",  "agent/codex/7.py"),
    ("codex8",  "agent/codex/8.py"),
    ("codex13", "agent/codex/13.py"),
    ("codex15", "agent/codex/15.py"),
    ("claude2", "agent/claude/2.py"),
    ("codex16", "agent/codex/16/agent.py"),
]

SEEDS = [42, 137, 256, 789]
NUM_WORKERS = 4
MAX_STEPS = 500

def setup_path():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

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
        def tb(i):
            s = env.players[i].stats
            return (s['kills'], s['boxes'], s['items'], s['bombs'])
        ss = sorted(survivors, key=tb, reverse=True)
        rv = 0
        for idx, pi in enumerate(ss):
            if idx > 0 and tb(pi) < tb(ss[idx-1]):
                rv = idx
            ranks[pi] = rv
    nsr = max(ranks[i] for i in survivors) + 1
    cr = nsr
    for j in reversed(death_order):
        ranks[j] = cr; cr += 1
    return ranks

def run_match(agents, seed, max_steps):
    from engine.game import BomberEnv
    env = BomberEnv(max_steps=max_steps)
    obs = env.reset(seed=seed)
    death_order = []
    prev_alive = [bool(p[2]) for p in obs["players"]]

    while True:
        actions = []
        for i in range(4):
            if int(obs["players"][i][2]) == 1:
                try:
                    actions.append(agents[i].act(obs))
                except Exception:
                    actions.append(0)
            else:
                actions.append(0)
        obs, term, trunc = env.step(actions)
        alive_now = [bool(p[2]) for p in obs["players"]]
        for i in range(4):
            if prev_alive[i] and not alive_now[i]:
                death_order.append(i)
        prev_alive = alive_now
        if term or trunc:
            break

    alive_final = [bool(p[2]) for p in obs["players"]]
    survivors = [i for i in range(4) if alive_final[i]]
    ranks = compute_ranks(survivors, death_order, env)
    stats = [env.players[i].stats.copy() for i in range(4)]
    return ranks, stats

def worker(work_items):
    setup_path()
    from competition.evaluation.runtime_guard import load_agent_instance

    # Group work items by unique combo so we load agents once per combo
    combo_items = {}
    for item in work_items:
        ci = item[0]
        combo_items.setdefault(ci, []).append(item)

    results = {}
    for ci, items in combo_items.items():
        # All items for this combo share the same agent_indices
        agent_indices = items[0][1]
        # Load 4 agents with correct game-position IDs (0-3)
        agents = []
        for pos, oi in enumerate(agent_indices):
            full = os.path.join(ROOT, AGENTS[oi][1])
            agents.append(load_agent_instance(full, pos))

        for _, _, seed, rotation in items:
            rotated = agents[-rotation:] + agents[:-rotation]
            ranks, stats = run_match(rotated, seed, MAX_STEPS)

            for pos_idx, orig_idx in enumerate(agent_indices):
                r_pos = (pos_idx + rotation) % 4
                results.setdefault(ci, {}).setdefault(orig_idx, {"ranks": [], "stats": []})
                results[ci][orig_idx]["ranks"].append(ranks[r_pos])
                results[ci][orig_idx]["stats"].append(stats[r_pos])

    return results

def main():
    n = len(AGENTS)
    combos = list(itertools.combinations(range(n), 4))
    total = len(combos) * len(SEEDS) * 4
    print(f"6 agents, {len(combos)} combos × {len(SEEDS)} seeds × 4 rotations = {total} matches")
    print(f"Workers: {NUM_WORKERS}")

    all_work = []
    for ci, combo in enumerate(combos):
        for seed in SEEDS:
            for rot in range(4):
                all_work.append((ci, list(combo), seed, rot))

    chunk_size = (total + NUM_WORKERS - 1) // NUM_WORKERS
    chunks = [all_work[i:i+chunk_size] for i in range(0, total, chunk_size)]

    start = time.time()
    all_results = {}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [pool.submit(worker, ch) for ch in chunks]
        for future in as_completed(futures):
            res = future.result()
            for ci, adata in res.items():
                if ci not in all_results:
                    all_results[ci] = adata
                else:
                    for oi, d in adata.items():
                        if oi not in all_results[ci]:
                            all_results[ci][oi] = d
                        else:
                            all_results[ci][oi]["ranks"].extend(d["ranks"])
                            all_results[ci][oi]["stats"].extend(d["stats"])
            done = sum(len(v) for v in all_results.values())
            print(f"  {done}/{total} matches done ({time.time()-start:.0f}s)")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/total:.2f}s/match)")

    # Aggregate per agent
    ar = {i: {"runs": 0, "ranks": [], "kills": [], "boxes": [], "items": [], "bombs": []} for i in range(n)}
    for ci, adata in all_results.items():
        for oi, d in adata.items():
            c = len(d["ranks"])
            ar[oi]["runs"] += c
            ar[oi]["ranks"].extend(d["ranks"])
            for s in d["stats"]:
                ar[oi]["kills"].append(s['kills'])
                ar[oi]["boxes"].append(s['boxes'])
                ar[oi]["items"].append(s['items'])
                ar[oi]["bombs"].append(s['bombs'])

    print(f"\n{'='*100}")
    hdr = f"{'Agent':<12} {'Win%':>8} {'AvgRank':>8} {'AvgKill':>8} {'AvgBox':>8} {'AvgItem':>8} {'AvgBomb':>8} {'Runs':>6}"
    print(hdr)
    print('-' * len(hdr))
    order = sorted(range(n), key=lambda i: -sum(1 for r in ar[i]["ranks"] if r == 0) / max(ar[i]["runs"], 1))
    for i in order:
        r = ar[i]
        nr = r["runs"]
        wp = sum(1 for rr in r["ranks"] if rr == 0) / nr * 100
        ak = sum(r["kills"]) / nr
        abo = sum(r["boxes"]) / nr
        ai = sum(r["items"]) / nr
        ab = sum(r["bombs"]) / nr
        arank = sum(r["ranks"]) / nr
        print(f"{AGENTS[i][0]:<12} {wp:>7.2f}% {arank:>8.3f} {ak:>7.2f} {abo:>7.1f} {ai:>7.2f} {ab:>7.2f} {nr:>6}")


if __name__ == "__main__":
    main()
