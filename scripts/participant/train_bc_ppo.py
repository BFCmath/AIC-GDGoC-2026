"""BC + PPO trainer for the hybrid Bomberland agent.

This script is the stable entrypoint intended for Colab:

    python scripts/participant/train_bc_ppo.py --mode full --device cuda

It keeps the training approach from ``docs/dl_approach.md`` practical:
behavior cloning from strong rule/search agents, PPO fine-tuning, and export
of a CPU-safe submission bundle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from agent import (
    BoxFarmerAgent,
    GeniusRuleAgent,
    RandomAgent,
    SimpleRuleAgent,
    SmarterRuleAgent,
    TacticalRuleAgent,
)
from competition.evaluation.runtime_guard import load_agent_instance, runtime_precheck
from engine.game import BomberEnv
from engine.map import Map
from engine.player import Player

ACTIONS = 6
BOARD = 13
BOMB_TIMER = 7
MAX_RADIUS = 5
SPATIAL_CHANNELS = 26
SCALAR_DIM = 13
MOVES = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

# Strong, diverse teachers. 4.py and 7.py remain important, but the training
# pool deliberately includes later/earlier Codex variants and rule baselines so
# PPO does not overfit to one local opponent style.
EXPERT_SPECS = [
    "agent/codex/8.py",
    "agent/codex/7.py",
    "agent/codex/6.py",
    "agent/codex/5.py",
    "agent/codex/4.py",
    "agent/codex/3.py",
    "agent/codex/2.py",
    "agent/codex/1.py",
    "TacticalRuleAgent",
    "GeniusRuleAgent",
    "SmarterRuleAgent",
    "BoxFarmerAgent",
]

OPPONENT_SPECS = [
    "agent/codex/8.py",
    "agent/codex/7.py",
    "agent/codex/6.py",
    "agent/codex/5.py",
    "agent/codex/4.py",
    "agent/codex/3.py",
    "agent/codex/2.py",
    "agent/codex/1.py",
    "TacticalRuleAgent",
    "GeniusRuleAgent",
    "SmarterRuleAgent",
    "BoxFarmerAgent",
    "SimpleRuleAgent",
    "RandomAgent",
]

OPPONENT_SCHEDULE = [
    {"agent": "RandomAgent", "strength": 0.10, "start_pct": 0.00, "peak_end_pct": 0.08, "end_pct": 0.35, "priority": 8, "label": "random"},
    {"agent": "SimpleRuleAgent", "strength": 0.20, "start_pct": 0.00, "peak_end_pct": 0.12, "end_pct": 0.45, "priority": 8, "label": "simple_rule"},
    {"agent": "BoxFarmerAgent", "strength": 0.30, "start_pct": 0.08, "peak_end_pct": 0.22, "end_pct": 0.60, "priority": 8, "label": "box_farmer"},
    {"agent": "SmarterRuleAgent", "strength": 0.40, "start_pct": 0.12, "peak_end_pct": 0.30, "end_pct": 0.72, "priority": 8, "label": "smarter_rule"},
    {"agent": "TacticalRuleAgent", "strength": 0.50, "start_pct": 0.20, "peak_end_pct": 0.45, "end_pct": 0.90, "priority": 8, "label": "tactical_rule"},
    {"agent": "agent/codex/4.py", "strength": 0.70, "start_pct": 0.25, "peak_end_pct": 0.55, "end_pct": 1.00, "priority": 10, "label": "codex4"},
    {"agent": "agent/codex/8.py", "strength": 0.85, "start_pct": 0.35, "peak_end_pct": 0.65, "end_pct": 1.00, "priority": 10, "label": "codex8"},
    {"agent": "agent/codex/7.py", "strength": 1.00, "start_pct": 0.45, "peak_end_pct": 0.75, "end_pct": 1.00, "priority": 10, "label": "codex7"},
]

DEFAULT_REWARD_CONFIG = {
    "rank_rewards": [15.0, 0.0, -10.0, -10.0],
    "win_strength_offset": 0.5,
    "loss_penalty_mult": 0.75,
    "strength_weighting": True,
}

EVAL_OPPONENTS = ["agent/codex/4.py", "agent/codex/7.py", "agent/codex/8.py"]

EXPERT_SAMPLE_POOL = (
    ["agent/codex/8.py"] * 5
    + ["agent/codex/7.py"] * 5
    + ["agent/codex/4.py"] * 4
    + ["agent/codex/6.py"] * 3
    + ["agent/codex/5.py"] * 2
    + ["agent/codex/3.py", "agent/codex/2.py", "agent/codex/1.py"]
    + ["TacticalRuleAgent", "GeniusRuleAgent", "SmarterRuleAgent", "BoxFarmerAgent"]
)



def compute_priority(entry: dict, pct: float) -> float:
    if pct < entry["start_pct"] or pct >= entry["end_pct"]:
        return 0.0
    if pct <= entry["peak_end_pct"]:
        return float(entry["priority"])
    decay_range = entry["end_pct"] - entry["peak_end_pct"]
    if decay_range <= 0:
        return float(entry["priority"])
    progress = (pct - entry["peak_end_pct"]) / decay_range
    return float(entry["priority"]) * (1.0 - progress)


def build_opponent_dist(schedule: list, pct: float, win_counters: dict | None = None):
    entries = []
    total = 0.0
    for entry in schedule:
        pri = compute_priority(entry, pct)
        if pri > 0:
            if win_counters:
                stats = win_counters.get(entry["agent"], {"wins": 0, "total": 0})
                if stats["total"] >= 10:
                    wr = stats["wins"] / stats["total"]
                    if wr > 0.75:
                        pri *= 0.85
                    elif wr < 0.35:
                        pri *= 1.15
            if pri > 0:
                entries.append((entry["agent"], entry["strength"], pri, entry.get("label", "")))
                total += pri
    if not entries:
        entries = [(e["agent"], e["strength"], 1.0, e.get("label", "")) for e in schedule if e["start_pct"] <= pct < e["end_pct"]]
        total = sum(e[2] for e in entries) or 1
    return [(a, s, p / total, l) for (a, s, p, l) in entries]


def sample_opponent(dist: list) -> str:
    agents = [d[0] for d in dist]
    probs = [d[2] for d in dist]
    return random.choices(agents, weights=probs, k=1)[0]


def get_opponent_strength(schedule: list, agent_spec: str) -> float:
    for entry in schedule:
        if entry["agent"] == agent_spec:
            return entry["strength"]
    return 1.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agent_from_file(path: str | Path, agent_id: int):
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    spec = importlib.util.spec_from_file_location(f"expert_{path.stem}_{agent_id}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent(agent_id)


def make_builtin_agent(name: str, agent_id: int):
    mapping = {
        "RandomAgent": RandomAgent,
        "SimpleRuleAgent": SimpleRuleAgent,
        "SmarterRuleAgent": SmarterRuleAgent,
        "TacticalRuleAgent": TacticalRuleAgent,
        "GeniusRuleAgent": GeniusRuleAgent,
        "BoxFarmerAgent": BoxFarmerAgent,
    }
    return mapping[name](agent_id)


def make_agent(spec: str, agent_id: int):
    if spec.endswith(".py") or "/" in spec:
        return load_agent_from_file(spec, agent_id)
    return make_builtin_agent(spec, agent_id)


def in_bounds(grid, x, y):
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]


def passable(grid, x, y):
    return in_bounds(grid, x, y) and int(grid[x, y]) in (
        Map.GRASS,
        Map.ITEM_RADIUS,
        Map.ITEM_CAPACITY,
    )


def next_pos(pos, action):
    dx, dy = MOVES.get(int(action), (0, 0))
    return pos[0] + dx, pos[1] + dy


def bomb_positions(bombs):
    arr = np.asarray(bombs)
    return {(int(b[0]), int(b[1])) for b in arr.reshape(-1, 4)} if arr.size else set()


def bomb_radius(players, owner_id):
    owner_id = int(owner_id)
    if 0 <= owner_id < len(players):
        return max(1, min(MAX_RADIUS, 1 + int(players[owner_id][4])))
    return 2


def blast_tiles(grid, bx, by, radius):
    tiles = {(int(bx), int(by))}
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for step in range(1, int(radius) + 1):
            x, y = int(bx) + dx * step, int(by) + dy * step
            if not in_bounds(grid, x, y):
                break
            cell = int(grid[x, y])
            if cell == Map.WALL:
                break
            tiles.add((x, y))
            if cell == Map.BOX:
                break
    return tiles


def danger_schedule(grid, bombs, players, horizon=7, extra_bomb=None):
    schedule = {t: set() for t in range(1, horizon + 1)}
    bomb_list = []
    arr = np.asarray(bombs)
    if arr.size:
        for b in arr.reshape(-1, 4):
            bomb_list.append(
                {
                    "pos": (int(b[0]), int(b[1])),
                    "timer": max(1, int(b[2])),
                    "radius": bomb_radius(players, int(b[3])),
                }
            )
    if extra_bomb is not None:
        pos, radius, timer = extra_bomb
        bomb_list.append({"pos": tuple(pos), "timer": max(1, int(timer)), "radius": int(radius)})

    times = [b["timer"] for b in bomb_list]
    changed = True
    while changed:
        changed = False
        for i, bomb in enumerate(bomb_list):
            tiles = blast_tiles(grid, bomb["pos"][0], bomb["pos"][1], bomb["radius"])
            for j, other in enumerate(bomb_list):
                if i != j and other["pos"] in tiles and times[j] > times[i]:
                    times[j] = times[i]
                    changed = True

    for bomb, timer in zip(bomb_list, times):
        if 1 <= timer <= horizon:
            schedule[timer].update(blast_tiles(grid, bomb["pos"][0], bomb["pos"][1], bomb["radius"]))
    return schedule


def boxes_hit_if_bomb(grid, pos, radius):
    return sum(1 for x, y in blast_tiles(grid, pos[0], pos[1], radius) if int(grid[x, y]) == Map.BOX)


def enemies_hit_if_bomb(grid, players, agent_id, pos, radius):
    tiles = blast_tiles(grid, pos[0], pos[1], radius)
    return sum(
        1
        for i, player in enumerate(players)
        if i != agent_id and int(player[2]) == 1 and (int(player[0]), int(player[1])) in tiles
    )


def safe_bfs_action(obs, agent_id, horizon=8, extra_bomb=None):
    grid, players, bombs = obs["map"], obs["players"], obs["bombs"]
    start = (int(players[agent_id][0]), int(players[agent_id][1]))
    danger = danger_schedule(grid, bombs, players, horizon=horizon, extra_bomb=extra_bomb)
    blocked = bomb_positions(bombs)
    queue = deque([(start, 0, None)])
    seen = {(start, 0)}
    while queue:
        pos, tick, first = queue.popleft()
        future_bad = any(pos in danger.get(t, set()) for t in range(tick + 1, horizon + 1))
        if tick > 0 and not future_bad:
            return first if first is not None else 0
        if tick >= horizon:
            continue
        for action in [0, 1, 2, 3, 4]:
            npos = next_pos(pos, action)
            if not passable(grid, npos[0], npos[1]):
                continue
            if npos in blocked and npos != start:
                continue
            if npos in danger.get(tick + 1, set()):
                continue
            state = (npos, tick + 1)
            if state in seen:
                continue
            seen.add(state)
            queue.append((npos, tick + 1, action if first is None else first))
    return None


def can_escape_after_bomb(obs, agent_id):
    player = obs["players"][agent_id]
    pos = (int(player[0]), int(player[1]))
    radius = 1 + int(player[4])
    return safe_bfs_action(obs, agent_id, horizon=8, extra_bomb=(pos, radius, BOMB_TIMER)) is not None


def legal_action_mask(obs, agent_id, veto_bombs=True):
    grid, players, bombs = obs["map"], obs["players"], obs["bombs"]
    mask = np.zeros(ACTIONS, dtype=np.bool_)
    if agent_id >= len(players) or int(players[agent_id][2]) != 1:
        mask[0] = True
        return mask

    player = players[agent_id]
    pos = (int(player[0]), int(player[1]))
    blocked = bomb_positions(bombs)
    mask[0] = True
    for action in [1, 2, 3, 4]:
        npos = next_pos(pos, action)
        if passable(grid, npos[0], npos[1]) and npos not in blocked:
            mask[action] = True

    if int(player[3]) > 0 and pos not in blocked:
        mask[5] = True
        if veto_bombs:
            radius = 1 + int(player[4])
            useful = boxes_hit_if_bomb(grid, pos, radius) > 0 or enemies_hit_if_bomb(
                grid, players, agent_id, pos, radius
            ) > 0
            mask[5] = useful and can_escape_after_bomb(obs, agent_id)

    if not mask.any():
        mask[0] = True
    return mask


def forced_safety_action(obs, agent_id):
    if agent_id >= len(obs["players"]) or int(obs["players"][agent_id][2]) != 1:
        return 0
    pos = (int(obs["players"][agent_id][0]), int(obs["players"][agent_id][1]))
    danger = danger_schedule(obs["map"], obs["bombs"], obs["players"], horizon=7)
    if any(pos in danger[t] for t in danger):
        escape = safe_bfs_action(obs, agent_id, horizon=8)
        if escape is not None:
            return int(escape)
    return None


def nearest_distance(start, targets, default=13):
    if not targets:
        return float(default)
    return float(min(abs(start[0] - t[0]) + abs(start[1] - t[1]) for t in targets))


def reachable_cells(obs, agent_id, max_depth=3):
    grid, players, bombs = obs["map"], obs["players"], obs["bombs"]
    start = (int(players[agent_id][0]), int(players[agent_id][1]))
    blocked = bomb_positions(bombs)
    queue = deque([(start, 0)])
    seen = {start}
    depths = {start: 0}
    while queue:
        pos, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for action in [1, 2, 3, 4]:
            npos = next_pos(pos, action)
            if npos in seen or npos in blocked or not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            depths[npos] = depth + 1
            queue.append((npos, depth + 1))
    return depths


def encode_observation(obs, agent_id, step_count=0, max_steps=500):
    grid, players, bombs = obs["map"], obs["players"], obs["bombs"]
    height, width = grid.shape
    spatial = np.zeros((SPATIAL_CHANNELS, height, width), dtype=np.float32)
    spatial[0] = grid == Map.GRASS
    spatial[1] = grid == Map.WALL
    spatial[2] = grid == Map.BOX
    spatial[3] = grid == Map.ITEM_RADIUS
    spatial[4] = grid == Map.ITEM_CAPACITY

    player = players[agent_id]
    pos = (int(player[0]), int(player[1]))
    if int(player[2]) == 1:
        spatial[5, pos[0], pos[1]] = 1.0

    channel = 6
    for idx, other in enumerate(players):
        if idx != agent_id and int(other[2]) == 1 and channel <= 8:
            spatial[channel, int(other[0]), int(other[1])] = 1.0
            channel += 1

    arr = np.asarray(bombs)
    if arr.size:
        for bomb in arr.reshape(-1, 4):
            bx, by, timer, owner = int(bomb[0]), int(bomb[1]), int(bomb[2]), int(bomb[3])
            spatial[9, bx, by] = 1.0
            spatial[10, bx, by] = float(timer) / BOMB_TIMER
            spatial[11, bx, by] = bomb_radius(players, owner) / MAX_RADIUS
            spatial[12 if owner == agent_id else 13, bx, by] = 1.0

    danger = danger_schedule(grid, bombs, players, horizon=7)
    for timer in range(1, 8):
        for dx, dy in danger[timer]:
            spatial[13 + timer, dx, dy] = 1.0

    depths = reachable_cells(obs, agent_id, max_depth=3)
    for cell, depth in depths.items():
        if depth <= 1:
            spatial[21, cell[0], cell[1]] = 1.0
        if depth <= 2:
            spatial[22, cell[0], cell[1]] = 1.0
        spatial[23, cell[0], cell[1]] = 1.0

    radius = 1 + int(player[4])
    for bx, by in blast_tiles(grid, pos[0], pos[1], radius):
        spatial[24, bx, by] = 1.0

    unsafe = set().union(*danger.values()) if danger else set()
    for sx, sy in set(depths) - unsafe:
        spatial[25, sx, sy] = 1.0

    enemies = [(int(p[0]), int(p[1])) for i, p in enumerate(players) if i != agent_id and int(p[2]) == 1]
    items = [(x, y) for x in range(height) for y in range(width) if int(grid[x, y]) in (Map.ITEM_RADIUS, Map.ITEM_CAPACITY)]
    box_spots = []
    for x in range(height):
        for y in range(width):
            if int(grid[x, y]) != Map.BOX:
                continue
            for action in [1, 2, 3, 4]:
                candidate = next_pos((x, y), action)
                if passable(grid, candidate[0], candidate[1]):
                    box_spots.append(candidate)

    current_danger = min([timer for timer in danger if pos in danger[timer]], default=0)
    boxes_hit = boxes_hit_if_bomb(grid, pos, radius)
    enemies_hit = enemies_hit_if_bomb(grid, players, agent_id, pos, radius)
    scalar = np.array(
        [
            agent_id / 3.0,
            min(float(step_count) / max_steps, 1.0),
            float(player[3]) / Player.MAX_BOMB_CAPACITY,
            float(radius) / MAX_RADIUS,
            len(enemies) / 3.0,
            nearest_distance(pos, enemies) / 24.0,
            nearest_distance(pos, items) / 24.0,
            nearest_distance(pos, box_spots) / 24.0,
            float(current_danger) / BOMB_TIMER,
            float(can_escape_after_bomb(obs, agent_id)) if int(player[3]) > 0 else 0.0,
            min(float(boxes_hit), 5.0) / 5.0,
            min(float(enemies_hit), 3.0) / 3.0,
            float(int(player[2]) == 1),
        ],
        dtype=np.float32,
    )
    return spatial, scalar


class PolicyValueNet(nn.Module):
    def __init__(self, spatial_channels=SPATIAL_CHANNELS, scalar_dim=SCALAR_DIM, num_actions=ACTIONS):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(nn.Linear(64 * BOARD * BOARD + 64, 256), nn.ReLU())
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, spatial, scalar):
        map_feat = self.map_encoder(spatial).flatten(1)
        scalar_feat = self.scalar_encoder(scalar)
        feat = self.trunk(torch.cat([map_feat, scalar_feat], dim=1))
        return self.policy(feat), self.value(feat).squeeze(-1)


def masked_logits(logits, masks):
    return logits.masked_fill(~masks.bool(), -1e9)


class ExpertDataset(Dataset):
    def __init__(self, spatial, scalar, masks, actions):
        self.spatial = torch.from_numpy(np.asarray(spatial, dtype=np.float32))
        self.scalar = torch.from_numpy(np.asarray(scalar, dtype=np.float32))
        self.masks = torch.from_numpy(np.asarray(masks, dtype=np.bool_))
        self.actions = torch.from_numpy(np.asarray(actions, dtype=np.int64))

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.spatial[idx], self.scalar[idx], self.masks[idx], self.actions[idx]


def get_symmetries(spatial, mask, action):
    # Action remap tables for the 8 dihedral group elements (D4)
    # actions: 0=STOP, 1=LEFT, 2=RIGHT, 3=UP, 4=DOWN, 5=BOMB
    remap = [
        [0, 1, 2, 3, 4, 5],  # 0: Identity
        [0, 4, 3, 1, 2, 5],  # 1: Rotate 90 CW (UP->RIGHT->DOWN->LEFT->UP)
        [0, 2, 1, 4, 3, 5],  # 2: Rotate 180 (UP<->DOWN, LEFT<->RIGHT)
        [0, 3, 4, 2, 1, 5],  # 3: Rotate 270 CW (UP->LEFT->DOWN->RIGHT->UP)
        [0, 1, 2, 4, 3, 5],  # 4: Flip Horizontal (LEFT<->RIGHT)
        [0, 2, 1, 3, 4, 5],  # 5: Flip Vertical (UP<->DOWN)
        [0, 3, 4, 1, 2, 5],  # 6: Flip Diagonal (Transpose, UP<->LEFT, DOWN<->RIGHT)
        [0, 4, 3, 2, 1, 5],  # 7: Flip anti-diagonal: (x,y)->(N-1-y,N-1-x)
    ]
    
    symmetries = []
    for sym_idx in range(8):
        if sym_idx == 0:
            sp = spatial
        elif sym_idx == 1:
            sp = np.rot90(spatial, k=-1, axes=(1, 2))
        elif sym_idx == 2:
            sp = np.rot90(spatial, k=-2, axes=(1, 2))
        elif sym_idx == 3:
            sp = np.rot90(spatial, k=-3, axes=(1, 2))
        elif sym_idx == 4:
            sp = np.flip(spatial, axis=2)
        elif sym_idx == 5:
            sp = np.flip(spatial, axis=1)
        elif sym_idx == 6:
            sp = np.transpose(spatial, axes=(0, 2, 1))
        elif sym_idx == 7:
            # True anti-diagonal reflection.  The previous implementation only
            # flipped one axis after transpose, which made the spatial transform
            # inconsistent with the action remap and injected wrong BC labels.
            sp = np.flip(np.transpose(spatial, axes=(0, 2, 1)), axis=(1, 2))
            
        m_table = remap[sym_idx]
        act_sym = m_table[action]
        mask_sym = np.zeros_like(mask)
        for a in range(6):
            mask_sym[m_table[a]] = mask[a]
            
        symmetries.append((sp, mask_sym, act_sym))
    return symmetries


def collect_expert_dataset(num_matches, max_steps, seed, accept_safe_trap_bombs=True):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    spatial_rows, scalar_rows, mask_rows, action_rows = [], [], [], []
    skipped = 0
    action_hist = {a: 0 for a in range(ACTIONS)}
    for match in range(num_matches):
        specs = [random.choice(EXPERT_SAMPLE_POOL) for _ in range(4)]
        experts = [make_agent(specs[i], i) for i in range(4)]
        obs = env.reset(seed=seed + match)
        for step in range(max_steps):
            actions = []
            for agent_id, expert in enumerate(experts):
                try:
                    action = int(expert.act(obs))
                except Exception:
                    action = 0
                action = action if 0 <= action < ACTIONS else 0
                actions.append(action)
                if int(obs["players"][agent_id][2]) != 1:
                    continue

                # BC should imitate strong agents' trap/farming bombs when they
                # are legal and escapable.  The previous veto_bombs=True mask
                # rejected every bomb that did not immediately hit a box/enemy,
                # which deletes many pressure/trap examples from codex 4/7/8.
                mask = legal_action_mask(obs, agent_id, veto_bombs=not accept_safe_trap_bombs)
                if accept_safe_trap_bombs:
                    mask = legal_action_mask(obs, agent_id, veto_bombs=False)
                    if action == 5 and not can_escape_after_bomb(obs, agent_id):
                        skipped += 1
                        continue
                if not mask[action]:
                    skipped += 1
                    continue
                spatial, scalar = encode_observation(obs, agent_id, step, max_steps=max_steps)

                # Apply 8-fold D4 symmetries data augmentation.
                for sp, msk, act in get_symmetries(spatial, mask, action):
                    spatial_rows.append(np.ascontiguousarray(sp))
                    scalar_rows.append(scalar)
                    mask_rows.append(msk)
                    action_rows.append(act)
                    action_hist[int(act)] += 1
            obs, terminated, truncated = env.step(actions)
            if terminated or truncated:
                break
        if (match + 1) % max(1, num_matches // 10) == 0:
            hist = ", ".join(f"{a}:{action_hist[a]}" for a in range(ACTIONS))
            print(f"BC collection {match + 1}/{num_matches}: {len(action_rows)} samples, skipped={skipped}, hist=[{hist}]")
    return ExpertDataset(spatial_rows, scalar_rows, mask_rows, action_rows)

def train_behavior_cloning(model, dataset, device, epochs, batch_size, lr, grad_clip, weighted_loss=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    action_weight = None
    if weighted_loss and len(dataset) > 0:
        counts = torch.bincount(dataset.actions, minlength=ACTIONS).float()
        inv = counts.sum() / torch.clamp(counts, min=1.0)
        # Keep the correction mild: enough to prevent STOP/move dominance, not
        # enough to make rare bombs explode in probability.
        action_weight = torch.sqrt(inv / inv.mean()).to(device)
        print("BC action weights:", [round(float(x), 3) for x in action_weight.cpu()])
    model.train()
    best_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        total_loss, total_correct, total_seen = 0.0, 0, 0
        for spatial, scalar, masks, actions in loader:
            spatial, scalar = spatial.to(device), scalar.to(device)
            masks, actions = masks.to(device), actions.to(device)
            logits, _ = model(spatial, scalar)
            logits = masked_logits(logits, masks)
            loss = F.cross_entropy(logits, actions, weight=action_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += float(loss.item()) * len(actions)
            total_correct += int((logits.argmax(dim=-1) == actions).sum().item())
            total_seen += len(actions)
        acc = total_correct / max(total_seen, 1)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"BC epoch {epoch + 1}/{epochs}: loss={total_loss / max(total_seen, 1):.4f}, acc={acc:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best BC epoch by train accuracy: acc={best_acc:.3f}")

def sample_masked_action(model, spatial, scalar, mask, deterministic=False):
    logits, value = model(spatial, scalar)
    logits = masked_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=logits)
    action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
    return action, dist.log_prob(action), dist.entropy(), value


def rank_players(env, death_steps=None):
    """BTC-accurate ranking.
    
    Rules (from BTC benchmark):
      - Dead earliest → worst rank.
      - Dead same step → same rank among those dead.
      - Survivors after max steps → tie-break by kills → boxes → items → bombs.
      - Survivors always rank better than dead players.
    
    Args:
        env: The BomberEnv instance.
        death_steps: dict mapping player_id → step when they died.
                     If None, falls back to arbitrary dead ordering (legacy).
    """
    if death_steps is None:
        death_steps = {}
    
    survivors = [i for i, p in enumerate(env.players) if p.alive]
    dead = [i for i in range(4) if i not in survivors]
    
    stats_key = lambda i: (
        env.players[i].stats["kills"],
        env.players[i].stats["boxes"],
        env.players[i].stats["items"],
        env.players[i].stats["bombs"],
    )
    
    ranks = [0] * 4
    
    # --- Rank survivors by stats tie-break (best stats = rank 0) ---
    if survivors:
        ordered = sorted(survivors, key=stats_key, reverse=True)
        rank = 0
        for idx, pid in enumerate(ordered):
            if idx > 0 and stats_key(pid) < stats_key(ordered[idx - 1]):
                rank = idx
            ranks[pid] = rank
    
    # --- Rank dead players by death step (died later = better rank) ---
    if dead:
        # Base rank for dead: one worse than worst survivor
        base = max((ranks[i] for i in survivors), default=-1) + 1
        # Sort dead by death_step descending (died later → ranked better)
        dead_sorted = sorted(dead, key=lambda i: death_steps.get(i, 0), reverse=True)
        rank = base
        for idx, pid in enumerate(dead_sorted):
            if idx > 0 and death_steps.get(pid, 0) < death_steps.get(dead_sorted[idx - 1], 0):
                rank = base + idx
            ranks[pid] = rank
    
    return ranks


def is_death_by_own_bomb(prev_obs, prev_pos, agent_id):
    bombs = prev_obs["bombs"]
    grid = prev_obs["map"]
    players = prev_obs["players"]
    for b in np.asarray(bombs).reshape(-1, 4):
        bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        if owner == agent_id and timer == 1:
            radius = bomb_radius(players, owner)
            blast = blast_tiles(grid, bx, by, radius)
            if prev_pos in blast:
                return True
    return False


def _get_safe_actions_mask(obs, aid):
    grid, players, bombs = obs["map"], obs["players"], obs["bombs"]
    mask = np.zeros(ACTIONS, dtype=np.bool_)
    if aid >= len(players) or int(players[aid][2]) != 1:
        mask[0] = True
        return mask

    player = players[aid]
    pos = (int(player[0]), int(player[1]))
    blocked = bomb_positions(bombs)
    
    danger = danger_schedule(grid, bombs, players, horizon=8)
    
    legal_actions = []
    for a in range(6):
        if a == 5:
            if int(player[3]) > 0 and pos not in blocked:
                legal_actions.append(a)
        else:
            npos = next_pos(pos, a)
            if passable(grid, npos[0], npos[1]) and (npos not in blocked or a == 0):
                legal_actions.append(a)

    has_escape = []
    no_escape = []
    for a in legal_actions:
        if a == 5:
            npos = pos
            extra = (pos, 1 + int(player[4]), BOMB_TIMER)
        else:
            npos = next_pos(pos, a)
            extra = None
            
        if npos in danger.get(1, set()):
            continue
            
        if safe_bfs_action(obs, aid, horizon=8, extra_bomb=extra) is not None:
            has_escape.append(a)
        else:
            no_escape.append(a)

    if has_escape:
        for a in has_escape:
            mask[a] = True
    elif no_escape:
        for a in no_escape:
            mask[a] = True
    else:
        for a in legal_actions:
            mask[a] = True

    if not mask.any():
        mask[0] = True
    return mask


def shaped_reward(prev_obs, obs, env, agent_id, action, done, prev_stats=None, cur_stats=None, stage=0, prev_visited_positions=None, death_steps=None, opponent_strengths=None, reward_config=None):
    if prev_obs is None:
        return 0.0
    prev_p = prev_obs["players"][agent_id]
    cur_p = obs["players"][agent_id]
    
    # Step reward: -0.01 per step to encourage speed
    reward = -0.01 if int(prev_p[2]) == 1 else 0.0
    
    # Death penalty: -10.0
    if int(prev_p[2]) == 1 and int(cur_p[2]) == 0:
        reward -= 10.0
        prev_pos = (int(prev_p[0]), int(prev_p[1]))
        if is_death_by_own_bomb(prev_obs, prev_pos, agent_id):
            reward -= 0.8

    if prev_stats is not None and cur_stats is not None:
        # kill opponent: +5.0 (and +10.0 bounty in boss stages 2-5)
        kills_diff = cur_stats["kills"] - prev_stats["kills"]
        if kills_diff > 0:
            reward += 5.0 * kills_diff
            if 2 <= stage <= 5:
                reward += 10.0 * kills_diff
            
        # destroy box: +0.5
        boxes_diff = cur_stats["boxes"] - prev_stats["boxes"]
        if boxes_diff > 0:
            reward += 0.5 * boxes_diff
            
        # collect item: +1.0
        items_diff = cur_stats["items"] - prev_stats["items"]
        if items_diff > 0:
            reward += 1.0 * items_diff

    # leave danger: +0.30
    prev_pos = (int(prev_p[0]), int(prev_p[1]))
    cur_pos = (int(cur_p[0]), int(cur_p[1]))
    
    # Wiggle penalty
    if action != 0 and prev_visited_positions is not None and cur_pos in prev_visited_positions:
        reward -= 0.05
        
    prev_danger = danger_schedule(prev_obs["map"], prev_obs["bombs"], prev_obs["players"], horizon=3)
    cur_danger = danger_schedule(obs["map"], obs["bombs"], obs["players"], horizon=3)
    
    in_prev_danger = any(prev_pos in prev_danger[t] for t in prev_danger)
    in_cur_danger = any(cur_pos in cur_danger[t] for t in cur_danger)
    
    if in_prev_danger and not in_cur_danger:
        reward += 0.3
        
    # enter unavoidable danger: -0.70
    if not in_prev_danger and in_cur_danger:
        reward -= 0.70
        
    # standing in future danger
    if cur_pos in cur_danger.get(1, set()) or cur_pos in cur_danger.get(2, set()):
        reward -= 0.2

    # place bomb value:
    if action == 5:
        has_escape = can_escape_after_bomb(prev_obs, agent_id)
        if not has_escape:
            reward -= 1.50
        else:
            bx, by = prev_pos
            radius = 1 + int(prev_p[4])
            boxes = boxes_hit_if_bomb(prev_obs["map"], prev_pos, radius)
            enemies = enemies_hit_if_bomb(prev_obs["map"], prev_obs["players"], agent_id, prev_pos, radius)
            if enemies > 0:
                reward += 0.80
            elif boxes > 0:
                reward += 0.25
                
    # useless STOP when safe
    if action == 0 and not in_prev_danger:
        reward -= 0.02

    # BTC tie-break shaping.  Most short/local curriculum games end at the step
    # limit, so the policy must learn kills > boxes > items > bombs rather than
    # pure survival.  This is deliberately small per step and larger at terminal.
    if prev_stats is not None and cur_stats is not None:
        bombs_diff = cur_stats["bombs"] - prev_stats["bombs"]
        if bombs_diff > 0 and action == 5:
            reward += 0.04 * bombs_diff

    # Terminal rank rewards
    if done:
        try:
            me = env.players[agent_id].stats
            opp_stats = [env.players[i].stats for i in range(4) if i != agent_id]
            stat_bonus = 0.0
            for opp in opp_stats:
                stat_bonus += 0.35 * np.sign(me["kills"] - opp["kills"])
                stat_bonus += 0.08 * np.sign(me["boxes"] - opp["boxes"])
                stat_bonus += 0.12 * np.sign(me["items"] - opp["items"])
                stat_bonus += 0.02 * np.sign(me["bombs"] - opp["bombs"])
            reward += float(stat_bonus)
        except Exception:
            pass
        
        ranks = rank_players(env, death_steps=death_steps)
        rank_val = ranks[agent_id]
        
        if reward_config is None:
            reward_config = DEFAULT_REWARD_CONFIG
        
        if opponent_strengths:
            avg_strength = sum(opponent_strengths) / len(opponent_strengths)
        else:
            avg_strength = 1.0
        
        winners = [i for i, r in enumerate(ranks) if r == 0]
        unique_win = rank_val == 0 and len(winners) == 1
        shared_best = rank_val == 0 and len(winners) > 1

        if unique_win:
            offset = reward_config.get("win_strength_offset", 0.5)
            terminal = reward_config["rank_rewards"][0] * (offset + avg_strength)
        elif shared_best:
            terminal = 2.0 * avg_strength
        elif rank_val == 1:
            terminal = reward_config["rank_rewards"][1]
        else:
            mult = reward_config.get("loss_penalty_mult", 0.75)
            terminal = reward_config["rank_rewards"][2] * (1.0 + mult * (1.0 - avg_strength))
        
        reward += terminal
            
    return float(reward)



@dataclass
class PPOBatch:
    spatial: list
    scalar: list
    masks: list
    actions: list
    logprobs: list
    rewards: list
    dones: list
    values: list
    last_value: float = 0.0

    def append(self, spatial, scalar, mask, action, logprob, reward, done, value):
        self.spatial.append(spatial)
        self.scalar.append(scalar)
        self.masks.append(mask)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    @classmethod
    def empty(cls):
        return cls([], [], [], [], [], [], [], [])


def make_snapshot_agent(snapshot_models, agent_id, device):
    if not snapshot_models:
        return None
    snapshot = random.choice(snapshot_models)
    # Snapshot opponents use the same hard safety wrapper but no live fallback to
    # avoid becoming just another copy of the rule teacher league.
    return NeuralSafeAgent(agent_id, snapshot, device, deterministic=False, fallback_agent=None, use_mask=True)


class NeuralSafeAgent:
    def __init__(self, agent_id, model, device, deterministic=False, fallback_agent=None, use_mask=True):
        self.agent_id = int(agent_id)
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.step_count = 0
        self.fallback_agent = fallback_agent
        self.use_mask = use_mask
        self.fallback_count = 0
        self.total_steps = 0

    def act(self, obs):
        self.total_steps += 1
        forced = forced_safety_action(obs, self.agent_id)
        if forced is not None and self.use_mask:
            return int(forced)
            
        mask = _get_safe_actions_mask(obs, self.agent_id)
        spatial, scalar = encode_observation(obs, self.agent_id, self.step_count)
        
        with torch.inference_mode():
            s = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
            a = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            
            logits, _ = self.model(s, a)
            probs = torch.softmax(logits, dim=-1)
            
            if self.use_mask:
                masked_log = masked_logits(logits, m)
                dist = torch.distributions.Categorical(logits=masked_log)
                action = torch.argmax(masked_log, dim=-1) if self.deterministic else dist.sample()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = torch.argmax(logits, dim=-1) if self.deterministic else dist.sample()
            
            action_i = int(action.item())
            prob = float(probs[0, action_i].item())
            if self.fallback_agent is not None:
                if prob < 0.25 or (self.use_mask and not bool(mask[action_i])):
                    try:
                        fallback_action = int(self.fallback_agent.act(obs))
                        self.fallback_count += 1
                        return fallback_action if 0 <= fallback_action < ACTIONS else 0
                    except Exception:
                        pass
                        
        self.step_count += 1
        return int(action.item())


def collect_ppo_rollout(model, device, horizon, envs, max_steps, seed, snapshot_models=None, update=0, total_updates=1, opponent_schedule=None, win_counters=None, reward_config=None):
    if opponent_schedule is None:
        opponent_schedule = OPPONENT_SCHEDULE
    trajectories = []
    model.eval()
    snapshot_models = snapshot_models or deque()
    pct = update / max(1, total_updates)
    snap_prob = min(0.35, 0.4 * pct)
    opp_dist = build_opponent_dist(opponent_schedule, pct, win_counters)
    
    for env_idx in range(envs):
        batch = PPOBatch.empty()
        control_id = random.randrange(4)
        env = BomberEnv(max_steps=max_steps, seed=seed + env_idx)
        obs = env.reset(seed=seed + env_idx)
        
        agents = []
        episode_opponent_types = []
        for i in range(4):
            if i == control_id:
                agents.append(NeuralSafeAgent(i, model, device, fallback_agent=make_agent("agent/codex/7.py", i)))
            else:
                snap_agent = make_snapshot_agent(snapshot_models, i, device) if random.random() < snap_prob else None
                if snap_agent is not None:
                    agents.append(snap_agent)
                else:
                    opp_name = sample_opponent(opp_dist)
                    agents.append(make_agent(opp_name, i))
                    episode_opponent_types.append(opp_name)
        
        death_steps = {}
        prev_alive = {i: True for i in range(4)}
        prev_visited_positions = deque(maxlen=4)
        
        for step in range(horizon):
            actions = []
            record = None
            for i, agent in enumerate(agents):
                if i == control_id:
                    forced = forced_safety_action(obs, i)
                    if forced is not None:
                        action = int(forced)
                        record = None
                    else:
                        spatial, scalar = encode_observation(obs, i, step, max_steps=max_steps)
                        mask = _get_safe_actions_mask(obs, i)
                        with torch.no_grad():
                            action_t, logprob_t, _, value_t = sample_masked_action(
                                model,
                                torch.from_numpy(spatial).unsqueeze(0).to(device),
                                torch.from_numpy(scalar).unsqueeze(0).to(device),
                                torch.from_numpy(mask).unsqueeze(0).to(device),
                            )
                        action = int(action_t.item())
                        record = (spatial, scalar, mask, action, float(logprob_t.item()), float(value_t.item()))
                else:
                    try:
                        action = int(agent.act(obs))
                    except Exception:
                        action = 0
                actions.append(action if 0 <= action < ACTIONS else 0)
                
            prev_stats = None
            if env.players[control_id] is not None:
                prev_stats = env.players[control_id].stats.copy()
            prev_obs_for_reward = obs

            next_obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            
            for i in range(4):
                if prev_alive[i] and not env.players[i].alive:
                    death_steps[i] = step
                    prev_alive[i] = False
            
            cur_stats = None
            if env.players[control_id] is not None:
                cur_stats = env.players[control_id].stats.copy()
            
            opponent_strengths = [get_opponent_strength(opponent_schedule, opp) for opp in episode_opponent_types]
            reward = shaped_reward(
                prev_obs_for_reward, next_obs, env, control_id, actions[control_id], done,
                prev_stats, cur_stats, stage=0,
                prev_visited_positions=prev_visited_positions,
                death_steps=death_steps,
                opponent_strengths=opponent_strengths,
                reward_config=reward_config,
            )
            
            if env.players[control_id] is not None and getattr(env.players[control_id], "alive", False):
                prev_visited_positions.append((int(env.players[control_id].x), int(env.players[control_id].y)))
                
            if record is not None:
                spatial, scalar, mask, action, logprob, value = record
                batch.append(spatial, scalar, mask, action, logprob, reward, done, value)
                
            obs = next_obs
            
            if done:
                break
        
        if batch.actions and not done:
            with torch.no_grad():
                s_last, a_last = encode_observation(obs, control_id, step, max_steps=max_steps)
                s_t = torch.from_numpy(s_last).unsqueeze(0).to(device)
                a_t = torch.from_numpy(a_last).unsqueeze(0).to(device)
                _, v_last = model(s_t, a_t)
                batch.last_value = float(v_last.item())
        
        if win_counters is not None and batch.actions:
            ranks = rank_players(env, death_steps=death_steps)
            if ranks[control_id] == 0:
                for opp in episode_opponent_types:
                    win_counters[opp]["wins"] += 1
            for opp in episode_opponent_types:
                win_counters[opp]["total"] += 1
        
        if batch.actions:
            trajectories.append(batch)
    return trajectories


def compute_gae(rewards, dones, values, gamma, gae_lambda, last_value=0.0):
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    next_value = float(last_value)
    for tick in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[tick]
        delta = rewards[tick] + gamma * next_value * nonterminal - values[tick]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[tick] = last_gae
        next_value = values[tick]
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def evaluate_agent_win_rate(model, device, max_steps, num_matches=20, agent_type="fallback", eval_opponents=None):
    if eval_opponents is None:
        eval_opponents = EVAL_OPPONENTS
    wins = 0
    total_fallback = 0
    total_steps = 0
    model.eval()
    for match in range(num_matches):
        env = BomberEnv(max_steps=max_steps, seed=1000000 + match)
        obs = env.reset(seed=1000000 + match)
        
        agents = []
        if agent_type == "neural":
            agent = NeuralSafeAgent(0, model, device, deterministic=True, fallback_agent=None, use_mask=False)
        elif agent_type == "mask":
            agent = NeuralSafeAgent(0, model, device, deterministic=True, fallback_agent=None, use_mask=True)
        elif agent_type == "codex7":
            agent = make_agent("agent/codex/7.py", 0)
        else: # "fallback"
            agent = NeuralSafeAgent(0, model, device, deterministic=True, fallback_agent=make_agent("agent/codex/7.py", 0), use_mask=True)
            
        agents.append(agent)
        
        for i in range(1, 4):
            agents.append(make_agent(random.choice(eval_opponents), i))
            
        death_steps = {}
        prev_alive = {i: True for i in range(4)}
        for step in range(max_steps):
            actions = []
            for i, agent in enumerate(agents):
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                actions.append(action if 0 <= action < ACTIONS else 0)
            obs, terminated, truncated = env.step(actions)
            # Track death steps for BTC-accurate ranking
            for i in range(4):
                if prev_alive[i] and not env.players[i].alive:
                    death_steps[i] = step
                    prev_alive[i] = False
            if terminated or truncated:
                break
                
        if hasattr(agents[0], "fallback_count"):
            total_fallback += agents[0].fallback_count
            total_steps += agents[0].total_steps
        else:
            total_steps += step + 1
            
        ranks = rank_players(env, death_steps=death_steps)
        if ranks[0] == 0:
            winners = [i for i in range(4) if ranks[i] == 0]
            if len(winners) == 1:
                wins += 1
                
    fallback_rate = total_fallback / max(1, total_steps)
    return float(wins) / num_matches, fallback_rate

def train_ppo(model, device, args):
    total_updates = max(1, args.ppo_updates)
    lr_start = args.ppo_lr_start if args.ppo_lr_start is not None else args.ppo_lr
    lr_end = args.ppo_lr_end if args.ppo_lr_end is not None else args.ppo_lr
    ent_start = args.ppo_ent_start if args.ppo_ent_start is not None else args.ent_coef
    ent_end = args.ppo_ent_end if args.ppo_ent_end is not None else args.ent_coef
    horizon_start = args.ppo_horizon_start if args.ppo_horizon_start is not None else args.ppo_horizon
    horizon_end = args.ppo_horizon_end if args.ppo_horizon_end is not None else args.ppo_horizon
    max_steps_start = args.max_steps_start if args.max_steps_start is not None else args.max_steps
    max_steps_end = args.max_steps_end if args.max_steps_end is not None else args.max_steps
    milestone_pcts = args.milestone_pcts if args.milestone_pcts is not None else [0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0]
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=1e-4)
    snapshot_models = deque(maxlen=24)
    
    initial_snap = PolicyValueNet().to(device)
    initial_snap.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    initial_snap.eval()
    snapshot_models.append(initial_snap)
    
    eval_interval = max(1, int(getattr(args, "eval_interval", 10)))
    eval_matches = max(1, int(getattr(args, "eval_matches", 12)))
    best_eval_score = -1e9
    milestone_dir = Path(getattr(args, "stage_checkpoint_dir", "checkpoints/milestones"))
    milestone_dir.mkdir(parents=True, exist_ok=True)
    saved_milestones = set()
    
    # Load opponent schedule and reward config
    opponent_schedule = list(OPPONENT_SCHEDULE)
    reward_config = dict(DEFAULT_REWARD_CONFIG)
    overrides_path = getattr(args, "overrides", "")
    if overrides_path:
        p = Path(overrides_path)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            data = json.loads(p.read_text())
            if "opponent_schedule" in data:
                opponent_schedule = data["opponent_schedule"]
                print("Applied opponent_schedule from overrides")
            if "reward" in data:
                reward_config.update(data["reward"])
                print("Applied reward config from overrides")
    
    win_counters = defaultdict(lambda: {"wins": 0, "total": 0})
    
    def run_eval(pct_label, update_label, force_milestones=False):
        nonlocal best_eval_score
        print(f"--- Evaluation at {pct_label*100:.1f}% (update {update_label}/{total_updates}) ---")
        wr_codex, _ = evaluate_agent_win_rate(model, device, current_max_steps, num_matches=eval_matches, agent_type="codex7")
        print(f"Codex 7         : Win Rate: {wr_codex*100:.1f}%")
        wr_neural, _ = evaluate_agent_win_rate(model, device, current_max_steps, num_matches=eval_matches, agent_type="neural")
        print(f"Neural Only     : Win Rate: {wr_neural*100:.1f}%")
        wr_mask, _ = evaluate_agent_win_rate(model, device, current_max_steps, num_matches=eval_matches, agent_type="mask")
        print(f"Neural + Mask   : Win Rate: {wr_mask*100:.1f}%")
        wr_fallback, fallback_rate = evaluate_agent_win_rate(model, device, current_max_steps, num_matches=eval_matches, agent_type="fallback")
        print(f"Neural+Mask+Fall: Win Rate: {wr_fallback*100:.1f}%, Fallback Rate: {fallback_rate*100:.1f}%")

        eval_score = max(wr_mask, wr_fallback) - 0.25 * fallback_rate
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            Path(args.best_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "pct": pct_label, "update": update_label, "eval_score": eval_score}, args.best_checkpoint)
            print(f"-> Saved best checkpoint (score={eval_score:.3f})")

        for mp in milestone_pcts:
            if (mp not in saved_milestones and pct_label >= mp - 0.005) or (force_milestones and mp not in saved_milestones and pct_label >= mp):
                saved_milestones.add(mp)
                ms_name = f"milestone_{mp*100:.0f}pct"
                ms_path = milestone_dir / f"{ms_name}_update{update_label}.pt"
                torch.save({"model_state_dict": model.state_dict(), "pct": pct_label, "update": update_label, "eval_score": eval_score}, ms_path)
                print(f"-> Saved milestone: {ms_path}")

    for update in range(args.ppo_updates):
        pct = update / total_updates
        
        # Scheduled hyperparameters (based on progress BEFORE this update)
        current_lr = lr_start + (lr_end - lr_start) * pct
        current_ent = ent_start + (ent_end - ent_start) * pct
        current_horizon = max(1, int(round(horizon_start + (horizon_end - horizon_start) * pct)))
        current_max_steps = max(1, int(round(max_steps_start + (max_steps_end - max_steps_start) * pct)))
        for pg in opt.param_groups:
            pg["lr"] = current_lr
        
        print(f"PPO {update+1}/{total_updates} | progress={pct*100:.1f}% | lr={current_lr:.2e} ent={current_ent:.4f} horizon={current_horizon} max_steps={current_max_steps}")
        
        trajectories = collect_ppo_rollout(
            model,
            device=device,
            horizon=current_horizon,
            envs=args.ppo_envs_per_update,
            max_steps=current_max_steps,
            seed=args.seed + 10000 * update,
            snapshot_models=snapshot_models,
            update=update,
            total_updates=total_updates,
            opponent_schedule=opponent_schedule,
            win_counters=win_counters,
            reward_config=reward_config,
        )
        if not trajectories:
            print(f"PPO {update + 1}/{total_updates}: empty rollout")
            continue

        all_advantages = []
        all_returns = []
        flat_batch = PPOBatch.empty()
        
        for traj in trajectories:
            adv, ret = compute_gae(traj.rewards, traj.dones, traj.values, args.gamma, args.gae_lambda, last_value=traj.last_value)
            all_advantages.append(adv)
            all_returns.append(ret)
            flat_batch.spatial.extend(traj.spatial)
            flat_batch.scalar.extend(traj.scalar)
            flat_batch.masks.extend(traj.masks)
            flat_batch.actions.extend(traj.actions)
            flat_batch.logprobs.extend(traj.logprobs)
            flat_batch.rewards.extend(traj.rewards)
            flat_batch.dones.extend(traj.dones)
            flat_batch.values.extend(traj.values)

        advantages = np.concatenate(all_advantages)
        returns = np.concatenate(all_returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        spatial = torch.from_numpy(np.asarray(flat_batch.spatial, dtype=np.float32)).to(device)
        scalar = torch.from_numpy(np.asarray(flat_batch.scalar, dtype=np.float32)).to(device)
        masks = torch.from_numpy(np.asarray(flat_batch.masks, dtype=np.bool_)).to(device)
        actions = torch.tensor(flat_batch.actions, dtype=torch.long, device=device)
        old_logprobs = torch.tensor(flat_batch.logprobs, dtype=torch.float32, device=device)
        advantages_t = torch.from_numpy(advantages).to(device)
        returns_t = torch.from_numpy(returns).to(device)
        indices = np.arange(len(actions))
        model.train()
        for _ in range(args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), args.ppo_minibatch_size):
                mb = indices[start : start + args.ppo_minibatch_size]
                logits, values = model(spatial[mb], scalar[mb])
                dist = torch.distributions.Categorical(logits=masked_logits(logits, masks[mb]))
                logprobs = dist.log_prob(actions[mb])
                ratio = (logprobs - old_logprobs[mb]).exp()
                pg1 = ratio * advantages_t[mb]
                pg2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantages_t[mb]
                policy_loss = -torch.min(pg1, pg2).mean()
                value_loss = F.mse_loss(values, returns_t[mb])
                loss = policy_loss + args.vf_coef * value_loss - current_ent * dist.entropy().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
        print(f"PPO {update+1}/{total_updates}: reward_mean={np.mean(flat_batch.rewards):.3f}, samples={len(actions)}")
        
        pct_after = (update + 1) / total_updates
        
        # Evaluation after PPO update (uses the just-trained model)
        if (update + 1) % eval_interval == 0:
            run_eval(pct_after, update + 1)
        
        # Save snapshot
        if (update + 1) % max(1, int(getattr(args, "snapshot_interval", 10))) == 0:
            snap = PolicyValueNet().to(device)
            snap.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
            snap.eval()
            snapshot_models.append(snap)
            snap_path = milestone_dir / f"snapshot_update{update+1}.pt"
            torch.save({"model_state_dict": model.state_dict(), "pct": pct_after, "update": update + 1}, snap_path)
            print(f"Saved snapshot: {snap_path}")
        
        # Adaptive priority adjustment
        if (update + 1) % max(1, eval_interval * 2) == 0:
            for entry in opponent_schedule:
                stats = win_counters.get(entry["agent"], {"wins": 0, "total": 0})
                if stats["total"] >= 10:
                    wr = stats["wins"] / stats["total"]
                    if wr > 0.75:
                        entry["priority"] = max(2, entry["priority"] * 0.85)
                        print(f"Adapt: reduced {entry.get('label', entry['agent'])} priority to {entry['priority']:.1f} (WR={wr:.2f})")
                    elif wr < 0.35:
                        entry["priority"] = min(20, entry["priority"] * 1.15)
                        print(f"Adapt: increased {entry.get('label', entry['agent'])} priority to {entry['priority']:.1f} (WR={wr:.2f})")

    # Force final evaluation + milestone at 100%
    run_eval(1.0, total_updates, force_milestones=True)


EXPORT_AGENT_TEMPLATE = """from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(1)

ACTIONS = 6
BOARD = 13
BOMB_TIMER = 7
MAX_RADIUS = 5
SPATIAL_CHANNELS = 26
SCALAR_DIM = 13
MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY = 0, 1, 2, 3, 4

class PolicyValueNet(nn.Module):
    def __init__(self, spatial_channels=SPATIAL_CHANNELS, scalar_dim=SCALAR_DIM, num_actions=ACTIONS):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 * BOARD * BOARD + 64, 256), nn.ReLU())
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)
    def forward(self, spatial, scalar):
        mf = self.map_encoder(spatial).flatten(1)
        sf = self.scalar_encoder(scalar)
        feat = self.trunk(torch.cat([mf, sf], dim=1))
        return self.policy(feat), self.value(feat).squeeze(-1)

def _inb(g, x, y): return 0 <= x < g.shape[0] and 0 <= y < g.shape[1]
def _pass(g, x, y): return _inb(g, x, y) and int(g[x, y]) in (GRASS, ITEM_RADIUS, ITEM_CAPACITY)
def _next(pos, a):
    dx, dy = MOVES.get(int(a), (0, 0)); return pos[0] + dx, pos[1] + dy
def _bomb_pos(bombs):
    arr = np.asarray(bombs)
    return {(int(b[0]), int(b[1])) for b in arr.reshape(-1, 4)} if arr.size else set()
def _radius(players, owner):
    owner = int(owner)
    return max(1, min(MAX_RADIUS, 1 + int(players[owner][4]))) if 0 <= owner < len(players) else 2
def _blast(g, bx, by, r):
    tiles = {(int(bx), int(by))}
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for k in range(1, int(r) + 1):
            x, y = int(bx) + dx * k, int(by) + dy * k
            if not _inb(g, x, y): break
            cell = int(g[x, y])
            if cell == WALL: break
            tiles.add((x, y))
            if cell == BOX: break
    return tiles
def _danger(g, bombs, players, horizon=7, extra=None):
    sched = {t: set() for t in range(1, horizon + 1)}
    bl = []
    arr = np.asarray(bombs)
    if arr.size:
        for b in arr.reshape(-1, 4):
            bl.append(((int(b[0]), int(b[1])), max(1, int(b[2])), _radius(players, int(b[3]))))
    if extra is not None:
        pos, r, timer = extra; bl.append((tuple(pos), max(1, int(timer)), int(r)))
    times = [b[1] for b in bl]
    changed = True
    while changed:
        changed = False
        for i, (pos, _, r) in enumerate(bl):
            bt = _blast(g, pos[0], pos[1], r)
            for j, (opos, _, _) in enumerate(bl):
                if i != j and opos in bt and times[j] > times[i]:
                    times[j] = times[i]; changed = True
    for (pos, _, r), t in zip(bl, times):
        if 1 <= t <= horizon:
            sched[t].update(_blast(g, pos[0], pos[1], r))
    return sched
def _safe_bfs(obs, aid, horizon=8, extra=None):
    g, players, bombs = obs['map'], obs['players'], obs['bombs']
    start = (int(players[aid][0]), int(players[aid][1]))
    danger = _danger(g, bombs, players, horizon, extra)
    blocked = _bomb_pos(bombs)
    q = deque([(start, 0, None)]); seen = {(start, 0)}
    while q:
        pos, t, first = q.popleft()
        if t > 0 and not any(pos in danger.get(tt, set()) for tt in range(t + 1, horizon + 1)):
            return first if first is not None else 0
        if t >= horizon: continue
        for a in [0, 1, 2, 3, 4]:
            np_ = _next(pos, a)
            if not _pass(g, np_[0], np_[1]): continue
            if np_ in blocked and np_ != start: continue
            if np_ in danger.get(t + 1, set()): continue
            st = (np_, t + 1)
            if st in seen: continue
            seen.add(st); q.append((np_, t + 1, a if first is None else first))
    return None
def _can_escape_bomb(obs, aid):
    p = obs['players'][aid]; pos = (int(p[0]), int(p[1])); r = 1 + int(p[4])
    return _safe_bfs(obs, aid, 8, (pos, r, BOMB_TIMER)) is not None
def _boxes_hit(g, pos, r): return sum(1 for x, y in _blast(g, pos[0], pos[1], r) if int(g[x, y]) == BOX)
def _enemies_hit(g, players, aid, pos, r):
    bt = _blast(g, pos[0], pos[1], r)
    return sum(1 for i, p in enumerate(players) if i != aid and int(p[2]) == 1 and (int(p[0]), int(p[1])) in bt)

def _get_safe_actions_mask(obs, aid):
    g, players, bombs = obs["map"], obs["players"], obs["bombs"]
    mask = np.zeros(ACTIONS, dtype=np.bool_)
    if aid >= len(players) or int(players[aid][2]) != 1:
        mask[0] = True
        return mask
    player = players[aid]
    pos = (int(player[0]), int(player[1]))
    blocked = _bomb_pos(bombs)
    danger = _danger(g, bombs, players, 8)
    legal_actions = []
    for a in range(6):
        if a == 5:
            if int(player[3]) > 0 and pos not in blocked:
                legal_actions.append(a)
        else:
            npos = _next(pos, a)
            if _pass(g, npos[0], npos[1]) and (npos not in blocked or a == 0):
                legal_actions.append(a)
    has_escape = []
    no_escape = []
    for a in legal_actions:
        if a == 5:
            npos = pos
            extra = (pos, 1 + int(player[4]), BOMB_TIMER)
        else:
            npos = _next(pos, a)
            extra = None
        if npos in danger.get(1, set()):
            continue
        if _safe_bfs(obs, aid, horizon=8, extra=extra) is not None:
            has_escape.append(a)
        else:
            no_escape.append(a)
    if has_escape:
        for a in has_escape:
            mask[a] = True
    elif no_escape:
        for a in no_escape:
            mask[a] = True
    else:
        for a in legal_actions:
            mask[a] = True
    if not mask.any():
        mask[0] = True
    return mask

def _nearest(start, targets, default=13):
    if not targets: return float(default)
    return float(min(abs(start[0] - t[0]) + abs(start[1] - t[1]) for t in targets))
def _reachable(obs, aid, depth=3):
    g, players, bombs = obs['map'], obs['players'], obs['bombs']
    start = (int(players[aid][0]), int(players[aid][1])); blocked = _bomb_pos(bombs)
    q = deque([(start, 0)]); seen = {start}; depths = {start: 0}
    while q:
        pos, d = q.popleft()
        if d >= depth: continue
        for a in [1, 2, 3, 4]:
            np_ = _next(pos, a)
            if np_ in seen or np_ in blocked or not _pass(g, np_[0], np_[1]): continue
            seen.add(np_); depths[np_] = d + 1; q.append((np_, d + 1))
    return depths
def _encode(obs, aid, step_count):
    g, players, bombs = obs['map'], obs['players'], obs['bombs']; h, w = g.shape
    x = np.zeros((SPATIAL_CHANNELS, h, w), dtype=np.float32)
    x[0] = (g == GRASS); x[1] = (g == WALL); x[2] = (g == BOX); x[3] = (g == ITEM_RADIUS); x[4] = (g == ITEM_CAPACITY)
    p = players[aid]; pos = (int(p[0]), int(p[1]))
    if int(p[2]) == 1: x[5, pos[0], pos[1]] = 1.0
    ch = 6
    for i, op in enumerate(players):
        if i != aid and int(op[2]) == 1 and ch <= 8:
            x[ch, int(op[0]), int(op[1])] = 1.0; ch += 1
    arr = np.asarray(bombs)
    if arr.size:
        for b in arr.reshape(-1, 4):
            bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            x[9, bx, by] = 1.0; x[10, bx, by] = float(timer) / BOMB_TIMER; x[11, bx, by] = _radius(players, owner) / MAX_RADIUS
            x[12 if owner == aid else 13, bx, by] = 1.0
    danger = _danger(g, bombs, players, 7)
    for t in range(1, 8):
        for dx, dy in danger[t]: x[13 + t, dx, dy] = 1.0
    depths = _reachable(obs, aid, 3)
    for rp, d in depths.items():
        if d <= 1: x[21, rp[0], rp[1]] = 1.0
        if d <= 2: x[22, rp[0], rp[1]] = 1.0
        x[23, rp[0], rp[1]] = 1.0
    r = 1 + int(p[4])
    for bx, by in _blast(g, pos[0], pos[1], r): x[24, bx, by] = 1.0
    unsafe = set().union(*danger.values()) if danger else set()
    for sx, sy in set(depths) - unsafe: x[25, sx, sy] = 1.0
    enemies = [(int(op[0]), int(op[1])) for i, op in enumerate(players) if i != aid and int(op[2]) == 1]
    items = [(i, j) for i in range(h) for j in range(w) if int(g[i, j]) in (ITEM_RADIUS, ITEM_CAPACITY)]
    box_spots = []
    for i in range(h):
        for j in range(w):
            if int(g[i, j]) == BOX:
                for a in [1, 2, 3, 4]:
                    bp = _next((i, j), a)
                    if _pass(g, bp[0], bp[1]): box_spots.append(bp)
    current_danger = min([t for t in danger if pos in danger[t]], default=0)
    scalar = np.array([aid / 3.0, min(step_count / 500.0, 1.0), float(p[3]) / 5.0, float(r) / MAX_RADIUS, len(enemies) / 3.0, _nearest(pos, enemies) / 24.0, _nearest(pos, items) / 24.0, _nearest(pos, box_spots) / 24.0, float(current_danger) / BOMB_TIMER, float(_can_escape_bomb(obs, aid)) if int(p[3]) > 0 else 0.0, min(float(_boxes_hit(g, pos, r)), 5.0) / 5.0, min(float(_enemies_hit(g, players, aid, pos, r)), 3.0) / 3.0, float(int(p[2]) == 1)], dtype=np.float32)
    return x, scalar
def _forced(obs, aid):
    if aid >= len(obs['players']) or int(obs['players'][aid][2]) != 1: return 0
    pos = (int(obs['players'][aid][0]), int(obs['players'][aid][1]))
    danger = _danger(obs['map'], obs['bombs'], obs['players'], 7)
    if any(pos in danger[t] for t in danger):
        esc = _safe_bfs(obs, aid, 8)
        if esc is not None: return int(esc)
    return None

class Agent:
    team_id = "LeagueBCPPO_Assassin"
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0
        self.device = torch.device('cpu')
        self.model = PolicyValueNet()
        ckpt = torch.load(Path(__file__).with_name('model.pt'), map_location='cpu')
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

        # Small teacher league used as a tactical fallback/reranker. The neural
        # policy only overrides this league when its safe action is clearly better.
        self.teacher_agents = []
        self.fallback_agent = None
        try:
            import importlib.util
            for fname in ("8.py", "7.py", "4.py"):
                fallback_path = Path(__file__).parent / fname
                if fallback_path.exists():
                    spec = importlib.util.spec_from_file_location(f"fallback_teacher_{fname}_{self.agent_id}", fallback_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    agent = module.Agent(self.agent_id)
                    # Prior reflects local ladder strength in the current repo: 8 is
                    # generally strongest, then 7/4.  The prior is only a tiny
                    # tie-break; the safety/tactical score still dominates.
                    prior = {"8.py": 0.10, "7.py": 0.05, "4.py": 0.00}.get(fname, 0.0)
                    self.teacher_agents.append((agent, prior))
                    if self.fallback_agent is None:
                        self.fallback_agent = agent
        except Exception:
            pass

    def _mobility_from(self, obs, start, depth=3):
        g, bombs = obs['map'], obs['bombs']
        blocked = _bomb_pos(bombs)
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if d >= depth:
                continue
            for a in [1, 2, 3, 4]:
                np_ = _next(pos, a)
                if np_ in seen or np_ in blocked or not _pass(g, np_[0], np_[1]):
                    continue
                seen.add(np_)
                q.append((np_, d + 1))
        return len(seen)

    def _box_spots(self, g):
        spots = []
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                if int(g[x, y]) != BOX:
                    continue
                for a in [1, 2, 3, 4]:
                    p = _next((x, y), a)
                    if _pass(g, p[0], p[1]):
                        spots.append(p)
        return spots

    def _action_score(self, obs, action):
        aid = self.agent_id
        g, players, bombs = obs['map'], obs['players'], obs['bombs']
        if aid >= len(players) or int(players[aid][2]) != 1:
            return -1e9 if action != 0 else 0.0
        mask = _get_safe_actions_mask(obs, aid)
        if action < 0 or action >= ACTIONS or not bool(mask[action]):
            return -1e9
        p = players[aid]
        pos = (int(p[0]), int(p[1]))
        radius = 1 + int(p[4])
        danger = _danger(g, bombs, players, 8)
        enemies = [(int(op[0]), int(op[1])) for i, op in enumerate(players) if i != aid and int(op[2]) == 1]
        items = [(x, y) for x in range(g.shape[0]) for y in range(g.shape[1]) if int(g[x, y]) in (ITEM_RADIUS, ITEM_CAPACITY)]
        box_spots = self._box_spots(g)

        if action == 5:
            if not _can_escape_bomb(obs, aid):
                return -1e8
            boxes = _boxes_hit(g, pos, radius)
            enemy_hits = _enemies_hit(g, players, aid, pos, radius)
            score = 0.15 + 1.25 * boxes + 7.0 * enemy_hits
            if boxes == 0 and enemy_hits == 0:
                score -= 1.5
            if pos in danger.get(1, set()):
                score -= 20.0
            return score

        npos = _next(pos, action)
        if not _pass(g, npos[0], npos[1]) or npos in _bomb_pos(bombs):
            return -1e9

        score = 0.0
        if npos in danger.get(1, set()):
            score -= 100.0
        if npos in danger.get(2, set()):
            score -= 7.0
        if any(npos in danger.get(t, set()) for t in range(3, 8)):
            score -= 0.35
        if pos in set().union(*danger.values()) and npos not in set().union(*danger.values()):
            score += 0.7

        cell = int(g[npos[0], npos[1]])
        if cell == ITEM_CAPACITY:
            score += 3.2
        elif cell == ITEM_RADIUS:
            score += 2.4

        score += 0.06 * self._mobility_from(obs, npos, 3)
        score += 0.16 * (_nearest(pos, items, 13) - _nearest(npos, items, 13))
        score += 0.12 * (_nearest(pos, box_spots, 13) - _nearest(npos, box_spots, 13))
        if radius >= 2 or self.step_count > 200:
            score += 0.05 * (_nearest(pos, enemies, 13) - _nearest(npos, enemies, 13))
        if action == 0:
            score -= 0.25
        return float(score)

    def _teacher_action(self, obs):
        best_action, best_score = 0, -1e18
        counts = {}
        for entry in self.teacher_agents:
            if isinstance(entry, tuple):
                teacher, prior = entry
            else:
                teacher, prior = entry, 0.0
            try:
                a = int(teacher.act(obs))
            except Exception:
                continue
            if not (0 <= a < ACTIONS):
                continue
            counts[a] = counts.get(a, 0) + 1
            score = self._action_score(obs, a) + prior + 0.04 * counts[a]
            if score > best_score:
                best_action, best_score = a, score
        if best_score <= -1e17 and self.fallback_agent is not None:
            try:
                return int(self.fallback_agent.act(obs))
            except Exception:
                return 0
        return best_action

    def act(self, obs):
        try:
            forced = _forced(obs, self.agent_id)
            if forced is not None:
                return int(forced)

            teacher_action = self._teacher_action(obs)
            teacher_score = self._action_score(obs, teacher_action)
            mask = _get_safe_actions_mask(obs, self.agent_id)
            spatial, scalar = _encode(obs, self.agent_id, self.step_count)

            with torch.inference_mode():
                s = torch.from_numpy(spatial).unsqueeze(0)
                a = torch.from_numpy(scalar).unsqueeze(0)
                logits, _ = self.model(s, a)
                probs = torch.softmax(logits, dim=-1)
                masked_log = logits.masked_fill(~torch.from_numpy(mask).unsqueeze(0).bool(), -1e9)
                neural_action = int(torch.argmax(masked_log, dim=-1).item())
                neural_prob = float(probs[0, neural_action].item())

            neural_score = self._action_score(obs, neural_action)
            self.step_count += 1
            # Conservative gate: until PPO is deeply trained, the neural policy is
            # mostly a tactical specialist.  Let it override the teacher league
            # only when both probability and independent tactical score agree.
            if neural_prob >= 0.72 and neural_score >= teacher_score + 0.35:
                return neural_action if 0 <= neural_action < ACTIONS else 0
            return teacher_action if 0 <= teacher_action < ACTIONS else 0
        except Exception:
            if self.fallback_agent is not None:
                try:
                    return int(self.fallback_agent.act(obs))
                except Exception:
                    return 0
            return 0
"""


def export_agent(model, export_dir):
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "spatial_channels": SPATIAL_CHANNELS,
            "scalar_dim": SCALAR_DIM,
            "actions": ACTIONS,
        },
        export_path / "model.pt",
    )
    (export_path / "agent.py").write_text(EXPORT_AGENT_TEMPLATE)
    
    # Copy top rule/search teachers as CPU-safe fallbacks.
    try:
        import shutil
        shutil.copy(ROOT / "agent/codex/8.py", export_path / "8.py")
        print(f"Copied 8.py as fallback teacher to {export_path}")
    except Exception as e:
        print(f"Warning: Failed to copy 8.py: {e}")

    # Copy 4.py as fallback teacher
    try:
        import shutil
        shutil.copy(ROOT / "agent/codex/4.py", export_path / "4.py")
        print(f"Copied 4.py as fallback teacher to {export_path}")
    except Exception as e:
        print(f"Warning: Failed to copy 4.py: {e}")
        
    # Copy 7.py as fallback teacher
    try:
        import shutil
        shutil.copy(ROOT / "agent/codex/7.py", export_path / "7.py")
        print(f"Copied 7.py as fallback teacher to {export_path}")
    except Exception as e:
        print(f"Warning: Failed to copy 7.py: {e}")

        
    print(f"Exported to {export_path}")
    return export_path


def smoke_test_export(export_dir, matches=2, max_steps=60):
    agent_file = Path(export_dir) / "agent.py"
    ok, message = runtime_precheck(str(agent_file), timeout_s=0.1, startup_timeout_s=3.0)
    print("Runtime precheck:", ok, message)
    env = BomberEnv(max_steps=max_steps, seed=123)
    agents = [load_agent_instance(str(agent_file), 0), TacticalRuleAgent(1), GeniusRuleAgent(2), SmarterRuleAgent(3)]
    survival = 0
    for match in range(matches):
        obs = env.reset(seed=123 + match)
        for _ in range(max_steps):
            actions = []
            for agent in agents:
                try:
                    actions.append(int(agent.act(obs)))
                except Exception:
                    actions.append(0)
            obs, terminated, truncated = env.step(actions)
            if terminated or truncated:
                break
        survival += int(bool(obs["players"][0][2]))
    print(f"Smoke survival: {survival}/{matches}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a hybrid BC + PPO Bomberland agent")
    parser.add_argument("--mode", choices=["bc", "ppo", "full", "export", "smoke"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--bc-matches", type=int, default=120)
    parser.add_argument("--bc-epochs", type=int, default=4)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-updates", type=int, default=80)
    parser.add_argument("--ppo-envs-per-update", type=int, default=16)
    parser.add_argument("--ppo-horizon", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=512)
    parser.add_argument("--ppo-lr", type=float, default=2.5e-4, help="Default PPO LR (used if start/end not set)")
    parser.add_argument("--ppo-lr-start", type=float, default=None, help="PPO LR at 0% training progress")
    parser.add_argument("--ppo-lr-end", type=float, default=None, help="PPO LR at 100% training progress")
    parser.add_argument("--ppo-ent-start", type=float, default=None, help="Entropy coef at 0% training progress")
    parser.add_argument("--ppo-ent-end", type=float, default=None, help="Entropy coef at 100% training progress")
    parser.add_argument("--ppo-horizon-start", type=int, default=None, help="PPO horizon at 0% training progress")
    parser.add_argument("--ppo-horizon-end", type=int, default=None, help="PPO horizon at 100% training progress")
    parser.add_argument("--max-steps-start", type=int, default=None, help="Episode max_steps at 0% training progress")
    parser.add_argument("--max-steps-end", type=int, default=None, help="Episode max_steps at 100% training progress")
    parser.add_argument("--eval-interval", type=int, default=10, help="PPO updates between evaluations")
    parser.add_argument("--eval-matches", type=int, default=12, help="Matches per PPO evaluation probe")
    parser.add_argument("--snapshot-interval", type=int, default=10, help="PPO updates between self-play snapshots")
    parser.add_argument("--stage-checkpoint-dir", default="checkpoints/milestones", help="Directory for milestone checkpoints")
    parser.add_argument("--best-checkpoint", default="checkpoints/best_hybrid_bc_ppo.pt", help="Best PPO checkpoint selected by eval score")
    parser.add_argument("--bc-unweighted", action="store_true", help="Disable mild inverse-frequency BC action weighting")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU torch threads; 1 is much faster in Colab/CPU smoke runs")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.015)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--overrides", type=str, default="", help="Path to overrides JSON (opponent_schedule, reward, etc.)")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-checkpoint", default="checkpoints/hybrid_bc_ppo.pt")
    parser.add_argument("--export-dir", default="exports/hybrid_ppo_agent")
    parser.add_argument("--smoke-matches", type=int, default=2)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if getattr(args, "torch_threads", 0):
        torch.set_num_threads(max(1, int(args.torch_threads)))
    
    if args.overrides:
        overrides_path = Path(args.overrides)
        if not overrides_path.is_absolute():
            overrides_path = ROOT / overrides_path
        if overrides_path.exists():
            overrides = json.loads(overrides_path.read_text())
            if "hyperparam" in overrides:
                hp = overrides["hyperparam"]
                for k, v in hp.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
                        print(f"Override hyperparam {k}={v}")
        else:
            print(f"Warning: overrides file {overrides_path} not found")
    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = PolicyValueNet().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if args.mode in {"bc", "full"}:
        dataset = collect_expert_dataset(args.bc_matches, args.max_steps, args.seed)
        train_behavior_cloning(model, dataset, device, args.bc_epochs, args.bc_batch_size, args.bc_lr, args.grad_clip, weighted_loss=not args.bc_unweighted)
        Path(args.save_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"Saved checkpoint to {args.save_checkpoint}")

    if args.mode in {"ppo", "full"}:
        train_ppo(model, device, args)
        Path(args.save_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"Saved checkpoint to {args.save_checkpoint}")

    if args.mode in {"export", "full", "smoke"}:
        export_agent(model, args.export_dir)

    if args.mode == "smoke":
        smoke_test_export(args.export_dir, matches=args.smoke_matches)


if __name__ == "__main__":
    main()
