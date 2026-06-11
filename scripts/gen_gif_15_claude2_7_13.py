import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from engine.game import BomberEnv
from competition.evaluation.runtime_guard import load_agent_instance
from competition.evaluation.rendering import render_match_frame

AGENTS = [
    ("agent/codex/15.py", "ResearchHybridV51"),
    ("agent/claude/2.py", "ApexHybridV2"),
    ("agent/codex/7.py", "AntiTrapHybridV9"),
    ("agent/codex/13.py", "ResearchHybridV30"),
]
SEED = 77

agents = []
for path, _ in AGENTS:
    a = load_agent_instance(str(Path(__file__).resolve().parents[1] / path), len(agents))
    agents.append(a)
    print(f"Loaded {path} -> {a.team_id}")

env = BomberEnv(max_steps=500)
obs = env.reset(seed=SEED)
frame_obs_list = []

while True:
    fo = {
        "map": obs["map"].tolist(),
        "players": obs["players"].tolist(),
        "bombs": obs["bombs"].tolist(),
        "_step": env.current_step,
    }
    frame_obs_list.append(fo)

    alive = sum(1 for p in obs["players"] if int(p[2]) == 1)
    if alive <= 1 or env.current_step >= 500:
        break

    actions = []
    for i in range(4):
        if int(obs["players"][i][2]) == 1:
            try: actions.append(agents[i].act(obs))
            except: actions.append(0)
        else: actions.append(0)
    obs, term, trunc = env.step(actions)
    if term or trunc:
        fo = {
            "map": obs["map"].tolist(),
            "players": obs["players"].tolist(),
            "bombs": obs["bombs"].tolist(),
            "_step": env.current_step,
        }
        frame_obs_list.append(fo)
        break

pil_frames = []
for i, fo in enumerate(frame_obs_list):
    prev = frame_obs_list[i-1] if i > 0 else None
    img = render_match_frame(fo, prev_obs=prev, agent_metadata={"agent_names": [a[1] for a in AGENTS]})
    pil_frames.append(img)

out = Path("codex15_claude2_v7_v13.gif")
pil_frames[0].save(str(out), save_all=True, append_images=pil_frames[1:], duration=120, loop=0)
print(f"Done! {len(pil_frames)} frames -> {out} ({out.stat().st_size/1e6:.1f}MB)")
