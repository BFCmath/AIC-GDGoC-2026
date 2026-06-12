"""
Lobby benchmark for a candidate agent vs the strong trio (codex/15, codex/7, claude/2).

Lobby types (candidate is always one of the "strong" seats):
  4S    : candidate + codex/15 + codex/7 + claude/2
  3S1W  : candidate + codex/15 + codex/7 + tactical
  2S2Wa : candidate + codex/15 + tactical + genius
  2S2Wb : candidate + codex/7  + tactical + genius

Each lobby: 4 corner rotations x N seeds. Reports candidate wins/draws/avg-rank
per lobby plus per-agent stats. Win = unique best rank (BTC rules incl.
truncation tie-break kills > boxes > items > bombs; dead ranked by death step).

Usage:
  python scripts/benchmark_lobbies.py --candidate agent/fable/1.py --seeds 8
"""

import argparse
import sys
import time
import concurrent.futures
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

STRONG_15 = "agent/codex/15.py"
STRONG_7 = "agent/codex/7.py"
STRONG_C2 = "agent/claude/2.py"
STRONG_13 = "agent/codex/13.py"
WEAK_TAC = "agent/tactical_rule_agent.py"
WEAK_GEN = "agent/genius_rule_agent.py"


def lobbies_for(candidate, full=False):
    lobbies = {
        "4S": [candidate, STRONG_15, STRONG_7, STRONG_C2],
        "3S1W": [candidate, STRONG_15, STRONG_7, WEAK_TAC],
        "2S2Wa": [candidate, STRONG_15, WEAK_TAC, WEAK_GEN],
        "2S2Wb": [candidate, STRONG_7, WEAK_TAC, WEAK_GEN],
    }
    if full:
        lobbies["4S-13"] = [candidate, STRONG_15, STRONG_13, STRONG_7]
        lobbies["3S1W-c2"] = [candidate, STRONG_C2, STRONG_15, WEAK_GEN]
        lobbies["2S2W-c2"] = [candidate, STRONG_C2, WEAK_TAC, WEAK_GEN]
    return lobbies


def rotate(lst, n):
    return lst[-n:] + lst[:-n]


def compute_ranks(survivors, death_steps, stats):
    """BTC-style ranks: survivors first by stats tiebreak, dead by death step."""
    ranks = [0] * 4
    if survivors:
        def tb_key(i):
            s = stats[i]
            return (s["kills"], s["boxes"], s["items"], s["bombs"])
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
    errors = defaultdict(int)
    while True:
        actions = []
        for s in range(4):
            if int(obs["players"][s][2]) == 1:
                try:
                    actions.append(agents[s].act(obs))
                except Exception:
                    errors[s] += 1
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
    ranks = compute_ranks(survivors, death_steps, stats)
    return {
        "lobby": lobby,
        "seat_order": seat_order,
        "seed": seed,
        "steps": step,
        "ranks": ranks,
        "survivors": survivors,
        "death_steps": death_steps,
        "stats": stats,
        "errors": dict(errors),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--full", action="store_true", help="add extra lobby variants")
    ap.add_argument("--lobby", default=None, help="run only this lobby key")
    args = ap.parse_args()

    lobbies = lobbies_for(args.candidate, args.full)
    if args.lobby:
        lobbies = {args.lobby: lobbies[args.lobby]}

    tasks = []
    for lobby, paths in lobbies.items():
        for corner in range(4):
            seat_order = rotate(list(range(4)), corner)
            for si in range(args.seeds):
                seed = args.base_seed + si * 17 + corner * 3
                tasks.append((lobby, paths, seat_order, seed))

    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_single_match, t) for t in tasks]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            results.append(f.result())
            if (i + 1) % 32 == 0:
                print(f"  ...{i+1}/{len(tasks)} matches", flush=True)
    print(f"\nRan {len(results)} matches in {time.time()-t0:.1f}s")

    cand_total = {"wins": 0, "draws": 0, "n": 0}
    for lobby, paths in lobbies.items():
        rs = [r for r in results if r["lobby"] == lobby]
        names = []
        for p in paths:
            pp = Path(p)
            names.append(pp.parent.name + "/" + pp.stem if pp.parent.name in ("codex", "claude", "fable") else pp.stem)
        agg = defaultdict(lambda: defaultdict(float))
        for r in rs:
            for seat in range(4):
                ai = r["seat_order"][seat]
                a = agg[ai]
                a["n"] += 1
                a["rank_sum"] += r["ranks"][seat]
                winners = [s for s in range(4) if r["ranks"][s] == 0]
                if r["ranks"][seat] == 0:
                    if len(winners) == 1:
                        a["wins"] += 1
                    else:
                        a["draws"] += 1
                if seat in r["survivors"]:
                    a["survived"] += 1
                st = r["stats"][seat]
                for k in ("kills", "boxes", "items", "bombs"):
                    a[k] += st[k]
                if r["errors"].get(seat):
                    a["errors"] += r["errors"][seat]
        print("=" * 96)
        print(f"LOBBY {lobby}  ({len(rs)} matches, avg {sum(r['steps'] for r in rs)/len(rs):.0f} steps)")
        print(f"  {'agent':<26}{'win':>5}{'draw':>5}{'avgRank':>9}{'surv%':>7}{'kills':>7}{'boxes':>7}{'items':>7}{'bombs':>7}{'err':>5}")
        for ai, name in enumerate(names):
            a = agg[ai]
            n = a["n"]
            mark = " <— candidate" if ai == 0 else ""
            print(f"  {name:<26}{int(a['wins']):>5}{int(a['draws']):>5}{a['rank_sum']/n:>9.2f}{100*a['survived']/n:>6.0f}%"
                  f"{a['kills']/n:>7.2f}{a['boxes']/n:>7.1f}{a['items']/n:>7.2f}{a['bombs']/n:>7.1f}{int(a['errors']):>5}{mark}")
        cand_total["wins"] += int(agg[0]["wins"])
        cand_total["draws"] += int(agg[0]["draws"])
        cand_total["n"] += int(agg[0]["n"])
    print("=" * 96)
    print(f"CANDIDATE TOTAL: {cand_total['wins']} wins, {cand_total['draws']} draws over {cand_total['n']} matches"
          f"  (win rate {100*cand_total['wins']/max(1,cand_total['n']):.0f}%)")


if __name__ == "__main__":
    main()
