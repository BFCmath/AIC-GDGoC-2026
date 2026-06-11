import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from engine.game import BomberEnv
from competition.evaluation.runtime_guard import load_agent_instance
from competition.evaluation.rendering import render_match_frame

AGENT_PATHS = [
    "agent/codex/7.py",
    "agent/codex/13.py",
    "agent/claude/2.py",
    "agent/codex/4.py",
]
TEAM_IDS = ["AntiTrapHybridV9", "ResearchHybridV30", "ApexHybridV2", "StatKillerHybridV4"]

SEEDS = [42, 137, 256, 789, 1024]

def obs_to_frame_obs(obs, step):
    return {
        "map": obs["map"].tolist(),
        "players": obs["players"].tolist(),
        "bombs": obs["bombs"].tolist(),
        "_step": step,
    }

def run_match(agents, seed, max_steps=500):
    env = BomberEnv(max_steps=max_steps)
    obs = env.reset(seed=seed)
    n_players = len(agents)
    frame_obs_list = []

    while True:
        frame_obs_list.append(obs_to_frame_obs(obs, env.current_step))
        if env.current_step >= max_steps:
            break

        alive_count = sum(1 for p in obs["players"] if int(p[2]) == 1)
        if alive_count <= 1:
            break

        actions = []
        for i in range(n_players):
            if int(obs["players"][i][2]) == 1:
                try:
                    action = agents[i].act(obs)
                except Exception as e:
                    print(f"Agent {i} error: {e}")
                    action = 0
                actions.append(action)
            else:
                actions.append(0)

        obs, terminated, truncated = env.step(actions)
        if terminated or truncated:
            frame_obs_list.append(obs_to_frame_obs(obs, env.current_step))
            break

    return frame_obs_list

def main():
    agents = []
    for path in AGENT_PATHS:
        full_path = str(Path(__file__).resolve().parents[1] / path)
        agent = load_agent_instance(full_path, len(agents))
        agents.append(agent)
        print(f"Loaded {path} -> {agents[-1].team_id}")

    out_dir = Path("gifs_output")
    out_dir.mkdir(exist_ok=True)

    for seed in SEEDS:
        print(f"\nRunning match with seed {seed}...")
        frames_obs = run_match(agents, seed)
        print(f"  {len(frames_obs)} frames captured")

        pil_frames = []
        for i, fo in enumerate(frames_obs):
            prev = frames_obs[i - 1] if i > 0 else None
            img = render_match_frame(fo, prev_obs=prev, agent_metadata={"agent_names": TEAM_IDS})
            pil_frames.append(img)

        gif_path = out_dir / f"match_seed{seed}.gif"
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=120,
            loop=0,
        )
        print(f"  Saved {gif_path} ({len(pil_frames)} frames)")

    print(f"\nDone! {len(SEEDS)} GIFs saved to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
