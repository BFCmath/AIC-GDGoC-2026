"""
MetaFarmSwitchV1 — greedy tie-break farmer with safety/anti-trap switch.

Primary objective: survive to timeout, then win tie-break by farming boxes/items/bombs.
Secondary objective: safe bomb pressure to deny/occasionally trap weak agents.
"""
import importlib.util
from pathlib import Path


def _load_local_agent(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent


FarmAgent = _load_local_agent("farm_impl.py", "_farm_impl")
SafeAgent = _load_local_agent("safe_impl.py", "_safe_impl")


class Agent:
    team_id = "MetaFarmSwitchV1"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.farm = FarmAgent(agent_id)   # aggressive stat farmer, c4-late style
        self.safe = SafeAgent(agent_id)   # anti-trap / box-tight safety style
        self.turn = 0
        self.risk_events = 0

    def act(self, obs: dict) -> int:
        self.turn += 1
        try:
            grid = obs["map"]
            players = obs["players"]
            bombs = obs["bombs"]
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                return 0

            me = players[self.agent_id]
            my_pos = (int(me[0]), int(me[1]))
            enemies = [
                (i, (int(p[0]), int(p[1])), 1 + int(p[4]), int(p[3]))
                for i, p in enumerate(players)
                if i != self.agent_id and int(p[2]) == 1
            ]

            # Use the safer child as a sensor for enemy potential bomb-line risk.
            self.safe._update_bomb_memory(bombs, players)
            risk = self.safe._enemy_future_bomb_risk(grid, my_pos, enemies, bombs)
            near = min(
                [abs(ep[0] - my_pos[0]) + abs(ep[1] - my_pos[1]) for _, ep, _, _ in enemies]
                or [99]
            )
            if risk >= 12 or near <= 2:
                self.risk_events += 1

            # If there is real bomb danger, never delegate to the greedier farmer.
            danger = self.safe._danger_schedule(grid, bombs, players, horizon=9)
            threatened = any(my_pos in danger.get(t, set()) for t in range(1, 8))
            if threatened or self.risk_events >= 3 or (self.turn > 180 and risk >= 8):
                return self.safe.act(obs)

            return self.farm.act(obs)
        except Exception:
            try:
                return self.safe.act(obs)
            except Exception:
                return 0
