"""
Probe analysis: how are wins decided in 2S2W / 3S1W / 4S lobbies?

For each lobby config, run matches over seeds x corner rotations and report:
  - per-agent: wins, draws, survival rate, avg death step, avg stats
  - how the match was decided: elimination (last alive) vs truncation tie-break,
    and which stat broke the tie (kills / boxes / items / bombs).
"""

import sys
import time
import concurrent.futures
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

LOBBIES = {
    "4S": ["agent/codex/15.py", "agent/codex/7.py", "agent/claude/2.py", "agent/codex/13.py"],
    "3S1W": ["agent/codex/15.py", "agent/codex/7.py", "agent/claude/2.py", "agent/tactical_rule_agent.py"],
    "2S2W": ["agent/codex/15.py", "agent/codex/7.py", "agent/tactical_rule_agent.py", "agent/genius_rule_agent.py"],
}

NUM_SEEDS = 6
NUM_CORNERS = 4
BASE_SEED = 1000


def rotate(lst, n):
    return lst[-n:] + lst[:-n]


def run_single_match(args):
    lobby, paths, seat_order, seed = args
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    from engine.game import BomberEnv
    from competition.evaluation.runtime_guard import load_agent_instance

    abs_paths = [str(ROOT / p) for p in paths]
    agents = [load_agent_instance(abs_paths[seat_order[s]], s) for s in range(4)]
    env = BomberEnv(max_steps=500)
    obs = env.reset(seed=seed)
    step = 0
    death_steps = {}
    prev_alive = [True] * 4
    while True:
        actions = []
        for s in range(4):
            if int(obs["players"][s][2]) == 1:
                try:
                    actions.append(agents[s].act(obs))
                except Exception:
                    actions.append(0)
            else:
                actions.append(0)
        obs, term, trunc = env.step(actions)
        step += 1
        alive_now = [bool(p[2]) for p in obs["players"]]
        for s in range(4):
            if prev_alive[s] and not alive_now[s]:
                death_steps[s] = step
        prev_alive = alive_now
        if term or trunc:
            break

    survivors = [s for s in range(4) if prev_alive[s]]
    stats = [dict(env.players[s].stats) for s in range(4)]

    # Determine winner(s) and decision mode
    decided_by = None
    if len(survivors) == 1:
        winners = survivors
        decided_by = "elimination"
    elif len(survivors) == 0:
        # all died; latest death wins
        mx = max(death_steps.values())
        winners = [s for s, d in death_steps.items() if d == mx]
        decided_by = "mutual_elimination"
    else:
        def key(s):
            st = stats[s]
            return (st["kills"], st["boxes"], st["items"], st["bombs"])
        best = max(key(s) for s in survivors)
        winners = [s for s in survivors if key(s) == best]
        # which stat broke the tie among survivors?
        ks = [key(s) for s in survivors]
        decided_by = "truncation:"
        for i, name in enumerate(["kills", "boxes", "items", "bombs"]):
            vals = {k[i] for k in ks}
            if len(vals) > 1:
                decided_by += name
                break
        else:
            decided_by += "fulldraw"

    return {
        "lobby": lobby,
        "seat_order": seat_order,
        "seed": seed,
        "steps": step,
        "survivors": survivors,
        "death_steps": death_steps,
        "stats": stats,
        "winners": winners,
        "decided_by": decided_by,
    }


def main():
    tasks = []
    for lobby, paths in LOBBIES.items():
        for corner in range(NUM_CORNERS):
            seat_order = rotate(list(range(4)), corner)
            for si in range(NUM_SEEDS):
                seed = BASE_SEED + si * 13 + corner
                tasks.append((lobby, paths, seat_order, seed))

    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
        for res in ex.map(run_single_match, tasks):
            results.append(res)
    print(f"Ran {len(results)} matches in {time.time()-t0:.1f}s\n")

    for lobby, paths in LOBBIES.items():
        rs = [r for r in results if r["lobby"] == lobby]
        names = [Path(p).parent.name + "/" + Path(p).stem for p in paths]
        agg = defaultdict(lambda: defaultdict(float))
        decided = defaultdict(int)
        for r in rs:
            decided[r["decided_by"]] += 1
            for seat in range(4):
                ai = r["seat_order"][seat]
                a = agg[ai]
                a["n"] += 1
                if seat in r["winners"]:
                    if len(r["winners"]) == 1:
                        a["wins"] += 1
                    else:
                        a["draws"] += 1
                if seat in r["survivors"]:
                    a["survived"] += 1
                else:
                    a["death_sum"] += r["death_steps"].get(seat, 0)
                    a["deaths"] += 1
                st = r["stats"][seat]
                for k in ("kills", "boxes", "items", "bombs"):
                    a[k] += st[k]
        print("=" * 90)
        print(f"LOBBY {lobby}: {names}   ({len(rs)} matches)")
        print(f"  decided_by: {dict(decided)}")
        print(f"  avg steps: {sum(r['steps'] for r in rs)/len(rs):.0f}")
        hdr = f"  {'agent':<28}{'win':>5}{'draw':>5}{'surv%':>7}{'avgdeath':>9}{'kills':>7}{'boxes':>7}{'items':>7}{'bombs':>7}"
        print(hdr)
        for ai, name in enumerate(names):
            a = agg[ai]
            n = a["n"]
            avg_death = a["death_sum"] / a["deaths"] if a["deaths"] else float("nan")
            print(f"  {name:<28}{int(a['wins']):>5}{int(a['draws']):>5}{100*a['survived']/n:>6.0f}%{avg_death:>9.0f}{a['kills']/n:>7.2f}{a['boxes']/n:>7.1f}{a['items']/n:>7.2f}{a['bombs']/n:>7.1f}")
        print()


if __name__ == "__main__":
    main()
