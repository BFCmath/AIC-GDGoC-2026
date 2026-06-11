"""codex15 + claude2 vs Smarter vs Tactical (10 eps)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.game import BomberEnv
from competition.evaluation.runtime_guard import load_agent_instance

AGENTS = [
    ("codex15", "agent/codex/15.py"),
    ("claude2",  "agent/claude/2.py"),
    ("Smarter",  "agent/smarter_rule_agent.py"),
    ("Tactical", "agent/tactical_rule_agent.py"),
]
NUM_EPS = 10; MAX_STEPS = 500; BASE_SEED = 42

def compute_ranks(survivors, death_order, env):
    ranks = [None]*4
    if len(survivors)==0:
        r=0
        for j in reversed(death_order): ranks[j]=r; r+=1
        return ranks
    if len(survivors)==1:
        ranks[survivors[0]]=0
    else:
        def tb(i): s=env.players[i].stats; return (s['kills'],s['boxes'],s['items'],s['bombs'])
        ss=sorted(survivors,key=tb,reverse=True)
        rv=0
        for idx,pi in enumerate(ss):
            if idx>0 and tb(pi)<tb(ss[idx-1]): rv=idx
            ranks[pi]=rv
    nsr=max(ranks[i] for i in survivors)+1; cr=nsr
    for j in reversed(death_order): ranks[j]=cr; cr+=1
    return ranks

agents=[]; labels=[]
for lb,path in AGENTS:
    a=load_agent_instance(str(Path(__file__).resolve().parents[1]/path),len(agents))
    agents.append(a); labels.append(lb)

tr=[0.]*4; wins=[0]*4; draws=[0]*4; tk=[0.]*4; tb=[0.]*4; ti=[0.]*4; tbo=[0.]*4

for ep in range(NUM_EPS):
    seed=BASE_SEED+ep*7
    env=BomberEnv(max_steps=MAX_STEPS)
    obs=env.reset(seed=seed)
    do=[]; pa=[bool(p[2]) for p in obs["players"]]
    while True:
        acts=[]
        for i in range(4):
            if int(obs["players"][i][2])==1:
                try: acts.append(agents[i].act(obs))
                except: acts.append(0)
            else: acts.append(0)
        obs,term,trunc=env.step(acts)
        an=[bool(p[2]) for p in obs["players"]]
        for i in range(4):
            if pa[i] and not an[i]: do.append(i)
        pa=an
        if term or trunc: break
    af=[bool(p[2]) for p in obs["players"]]
    sv=[i for i in range(4) if af[i]]
    ranks=compute_ranks(sv,do,env)
    ws=[i for i in range(4) if ranks[i]==0]
    for i in range(4):
        tr[i]+=ranks[i]
        if ranks[i]==0:
            if len(ws)==1: wins[i]+=1
            else: draws[i]+=1
        s=env.players[i].stats
        tk[i]+=s['kills']; tb[i]+=s['boxes']; ti[i]+=s['items']; tbo[i]+=s['bombs']
    wl=[labels[w] for w in ws]
    print(f"  Ep {ep+1:2d}: {' wins'.join(wl) if len(ws)==1 else 'Draw '+str(wl)} (seed={seed})")

n=NUM_EPS
print(f"\n{'Agent':<14} {'Win%':>7} {'Draw%':>7} {'AvgRank':>8} {'AvgKill':>8} {'AvgBox':>7} {'AvgItem':>8} {'AvgBomb':>8}")
print('-'*78)
for i in range(4):
    print(f"{labels[i]:<14} {wins[i]/n*100:>6.1f}% {draws[i]/n*100:>6.1f}% {tr[i]/n:>8.3f}  {tk[i]/n:>7.2f} {tb[i]/n:>6.1f} {ti[i]/n:>7.2f} {tbo[i]/n:>7.2f}")
