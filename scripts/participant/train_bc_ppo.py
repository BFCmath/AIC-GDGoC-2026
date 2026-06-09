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
import random
import sys
import time
from collections import deque
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

EXPERT_SPECS = [
    "agent/codex/7.py",
    "agent/codex/4.py",
    "agent/codex/1.py",
    "agent/codex/2.py",
    "TacticalRuleAgent",
    "GeniusRuleAgent",
    "SmarterRuleAgent",
    "BoxFarmerAgent",
]

OPPONENT_SPECS = [
    "agent/codex/7.py",
    "agent/codex/4.py",
    "agent/codex/1.py",
    "agent/codex/2.py",
    "TacticalRuleAgent",
    "GeniusRuleAgent",
    "SmarterRuleAgent",
    "BoxFarmerAgent",
    "SimpleRuleAgent",
]



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
    # actions: 0=STOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=BOMB
    remap = [
        [0, 1, 2, 3, 4, 5],  # 0: Identity
        [0, 4, 3, 1, 2, 5],  # 1: Rotate 90 CW (UP->RIGHT->DOWN->LEFT->UP)
        [0, 2, 1, 4, 3, 5],  # 2: Rotate 180 (UP<->DOWN, LEFT<->RIGHT)
        [0, 3, 4, 2, 1, 5],  # 3: Rotate 270 CW (UP->LEFT->DOWN->RIGHT->UP)
        [0, 1, 2, 4, 3, 5],  # 4: Flip Horizontal (LEFT<->RIGHT)
        [0, 2, 1, 3, 4, 5],  # 5: Flip Vertical (UP<->DOWN)
        [0, 3, 4, 1, 2, 5],  # 6: Flip Diagonal (Transpose, UP<->LEFT, DOWN<->RIGHT)
        [0, 4, 3, 2, 1, 5],  # 7: Flip Anti-diagonal (Transpose + Flip, UP<->RIGHT, DOWN<->LEFT)
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
            sp = np.flip(np.transpose(spatial, axes=(0, 2, 1)), axis=1)
            
        m_table = remap[sym_idx]
        act_sym = m_table[action]
        mask_sym = np.zeros_like(mask)
        for a in range(6):
            mask_sym[m_table[a]] = mask[a]
            
        symmetries.append((sp, mask_sym, act_sym))
    return symmetries


def collect_expert_dataset(num_matches, max_steps, seed):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    spatial_rows, scalar_rows, mask_rows, action_rows = [], [], [], []
    skipped = 0
    for match in range(num_matches):
        specs = []
        for i in range(4):
            r = random.random()
            if r < 0.45:
                specs.append("agent/codex/7.py")
            elif r < 0.75:
                specs.append("agent/codex/4.py")
            else:
                specs.append(random.choice(EXPERT_SPECS))

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
                mask = legal_action_mask(obs, agent_id, veto_bombs=True)
                if not mask[action]:
                    skipped += 1
                    continue
                spatial, scalar = encode_observation(obs, agent_id, step, max_steps=max_steps)
                
                # Apply 8-fold symmetries data augmentation
                for sp, msk, act in get_symmetries(spatial, mask, action):
                    spatial_rows.append(sp)
                    scalar_rows.append(scalar)
                    mask_rows.append(msk)
                    action_rows.append(act)
            obs, terminated, truncated = env.step(actions)
            if terminated or truncated:
                break
        if (match + 1) % max(1, num_matches // 10) == 0:
            print(f"BC collection {match + 1}/{num_matches}: {len(action_rows)} samples, skipped={skipped}")
    return ExpertDataset(spatial_rows, scalar_rows, mask_rows, action_rows)


def train_behavior_cloning(model, dataset, device, epochs, batch_size, lr, grad_clip):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        total_loss, total_correct, total_seen = 0.0, 0, 0
        for spatial, scalar, masks, actions in loader:
            spatial, scalar = spatial.to(device), scalar.to(device)
            masks, actions = masks.to(device), actions.to(device)
            logits, _ = model(spatial, scalar)
            logits = masked_logits(logits, masks)
            loss = F.cross_entropy(logits, actions)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += float(loss.item()) * len(actions)
            total_correct += int((logits.argmax(dim=-1) == actions).sum().item())
            total_seen += len(actions)
        print(f"BC epoch {epoch + 1}/{epochs}: loss={total_loss / max(total_seen, 1):.4f}, acc={total_correct / max(total_seen, 1):.3f}")


def sample_masked_action(model, spatial, scalar, mask, deterministic=False):
    logits, value = model(spatial, scalar)
    logits = masked_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=logits)
    action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
    return action, dist.log_prob(action), dist.entropy(), value


def rank_players(env):
    survivors = [i for i, p in enumerate(env.players) if p.alive]
    stats_key = lambda i: (
        env.players[i].stats["kills"],
        env.players[i].stats["boxes"],
        env.players[i].stats["items"],
        env.players[i].stats["bombs"],
    )
    ranks = [3] * 4
    if survivors:
        ordered = sorted(survivors, key=stats_key, reverse=True)
        rank = 0
        for idx, pid in enumerate(ordered):
            if idx > 0 and stats_key(pid) < stats_key(ordered[idx - 1]):
                rank = idx
            ranks[pid] = rank
    dead = [i for i in range(4) if i not in survivors]
    base = max([ranks[i] for i in survivors], default=-1) + 1
    for offset, pid in enumerate(dead):
        ranks[pid] = base + offset
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


def shaped_reward(prev_obs, obs, env, agent_id, action, done, prev_stats=None, cur_stats=None, stage=0, prev_visited_positions=None):
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
    capacity_diff = int(cur_p[3]) - int(prev_p[3])
    radius_diff = int(cur_p[4]) - int(prev_p[4])
    if capacity_diff > 0:
        reward += 1.0 * capacity_diff
    if radius_diff > 0:
        reward += 1.0 * radius_diff

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

    # Terminal rank rewards
    if done:
        ranks = rank_players(env)
        rank_val = ranks[agent_id]
        
        if stage <= 5:
            if rank_val == 0:
                reward += 15.0
            else:
                reward -= 10.0
        elif stage == 6:
            if rank_val == 0:
                reward += 15.0
            elif rank_val == 1:
                reward += 5.0
            else:
                reward -= 10.0
        else:  # stage 7
            ranks_reward = {0: 15.0, 1: 5.0, 2: -5.0, 3: -15.0}
            reward += ranks_reward.get(rank_val, -15.0)
            
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


class NeuralSafeAgent:
    def __init__(self, agent_id, model, device, deterministic=False, fallback_agent=None):
        self.agent_id = int(agent_id)
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.step_count = 0
        self.fallback_agent = fallback_agent

    def act(self, obs):
        forced = forced_safety_action(obs, self.agent_id)
        if forced is not None:
            return int(forced)
            
        mask = _get_safe_actions_mask(obs, self.agent_id)
        spatial, scalar = encode_observation(obs, self.agent_id, self.step_count)
        
        with torch.inference_mode():
            s = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
            a = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            
            logits, _ = self.model(s, a)
            probs = torch.softmax(logits, dim=-1)
            
            masked_log = masked_logits(logits, m)
            dist = torch.distributions.Categorical(logits=masked_log)
            action = torch.argmax(masked_log, dim=-1) if self.deterministic else dist.sample()
            
            prob = probs[0, action].item()
            if (prob < 0.25 or not mask[action]) and self.fallback_agent is not None:
                try:
                    return int(self.fallback_agent.act(obs))
                except Exception:
                    pass
                    
        self.step_count += 1
        return int(action.item())


def collect_ppo_rollout(model, device, horizon, envs, max_steps, seed, snapshot_models=None, stage=0):
    batch = PPOBatch.empty()
    model.eval()
    snapshot_models = snapshot_models or []
    
    for env_idx in range(envs):
        control_id = random.randrange(4)
        env = BomberEnv(max_steps=max_steps, seed=seed + env_idx)
        obs = env.reset(seed=seed + env_idx)
        
        # Build self-play/opponent pool based on stage
        agents = []
        for i in range(4):
            if i == control_id:
                agents.append(NeuralSafeAgent(i, model, device, fallback_agent=make_agent("agent/codex/7.py", i)))
            else:
                if stage == 0:
                    agents.append(make_agent(random.choice(["RandomAgent", "SimpleRuleAgent"]), i))
                elif stage == 1:
                    agents.append(make_agent(random.choice(["TacticalRuleAgent", "SmarterRuleAgent", "BoxFarmerAgent"]), i))
                elif stage == 2:
                    agents.append(make_agent(random.choice(["agent/codex/4.py", "SimpleRuleAgent", "SimpleRuleAgent"]), i))
                elif stage == 3:
                    agents.append(make_agent(random.choice(["agent/codex/8.py", "SimpleRuleAgent", "SimpleRuleAgent"]), i))
                elif stage == 4:
                    agents.append(make_agent(random.choice(["agent/codex/7.py", "SimpleRuleAgent", "SimpleRuleAgent"]), i))
                elif stage == 5:
                    agents.append(make_agent(random.choice(["agent/codex/7.py", "TacticalRuleAgent", "SmarterRuleAgent"]), i))
                elif stage == 6:
                    agents.append(make_agent(random.choice(["agent/codex/8.py", "agent/codex/4.py", "TacticalRuleAgent"]), i))
                else: # Stage 7
                    agents.append(make_agent(random.choice(["agent/codex/7.py", "agent/codex/4.py", "agent/codex/8.py"]), i))
                        
        prev_obs = None
        prev_stats = None
        prev_visited_positions = deque(maxlen=4)
        
        for step in range(horizon):
            actions = []
            record = None
            for i, agent in enumerate(agents):
                if i == control_id:
                    forced = forced_safety_action(obs, i)
                    spatial, scalar = encode_observation(obs, i, step, max_steps=max_steps)
                    mask = _get_safe_actions_mask(obs, i)
                    with torch.no_grad():
                        action_t, logprob_t, _, value_t = sample_masked_action(
                            model,
                            torch.from_numpy(spatial).unsqueeze(0).to(device),
                            torch.from_numpy(scalar).unsqueeze(0).to(device),
                            torch.from_numpy(mask).unsqueeze(0).to(device),
                        )
                    action = int(forced) if forced is not None else int(action_t.item())
                    record = (spatial, scalar, mask, action, float(logprob_t.item()), float(value_t.item()))
                else:
                    try:
                        action = int(agent.act(obs))
                    except Exception:
                        action = 0
                actions.append(action if 0 <= action < ACTIONS else 0)
                
            next_obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            
            cur_stats = None
            if env.players[control_id] is not None:
                cur_stats = env.players[control_id].stats.copy()
                
            reward = shaped_reward(prev_obs, next_obs, env, control_id, actions[control_id], done, prev_stats, cur_stats, stage=stage, prev_visited_positions=prev_visited_positions)
            
            if env.players[control_id] is not None and getattr(env.players[control_id], "alive", False):
                prev_visited_positions.append((int(env.players[control_id].x), int(env.players[control_id].y)))
                
            if record is not None:
                spatial, scalar, mask, action, logprob, value = record
                batch.append(spatial, scalar, mask, action, logprob, reward, done, value)
                
            prev_obs = obs
            obs = next_obs
            prev_stats = cur_stats
            
            if done:
                break
    return batch


def compute_gae(rewards, dones, values, gamma, gae_lambda):
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0
    for tick in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[tick]
        delta = rewards[tick] + gamma * next_value * nonterminal - values[tick]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[tick] = last_gae
        next_value = values[tick]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


def evaluate_agent_win_rate(model, device, stage, max_steps, num_matches=20):
    wins = 0
    model.eval()
    for match in range(num_matches):
        env = BomberEnv(max_steps=max_steps, seed=1000000 + stage * 1000 + match)
        obs = env.reset(seed=1000000 + stage * 1000 + match)
        
        agents = []
        agents.append(NeuralSafeAgent(0, model, device, deterministic=True, fallback_agent=make_agent("agent/codex/7.py", 0)))
        
        if stage == 0:
            opps = ["RandomAgent", "SimpleRuleAgent"]
        elif stage == 1:
            opps = ["TacticalRuleAgent", "SmarterRuleAgent", "BoxFarmerAgent"]
        elif stage == 2:
            opps = ["agent/codex/4.py", "SimpleRuleAgent", "SimpleRuleAgent"]
        elif stage == 3:
            opps = ["agent/codex/8.py", "SimpleRuleAgent", "SimpleRuleAgent"]
        elif stage == 4:
            opps = ["agent/codex/7.py", "SimpleRuleAgent", "SimpleRuleAgent"]
        elif stage == 5:
            opps = ["agent/codex/7.py", "TacticalRuleAgent", "SmarterRuleAgent"]
        elif stage == 6:
            opps = ["agent/codex/8.py", "agent/codex/4.py", "TacticalRuleAgent"]
        else: # stage 7
            opps = ["agent/codex/7.py", "agent/codex/4.py", "agent/codex/8.py"]
            
        for i in range(1, 4):
            agents.append(make_agent(random.choice(opps), i))
            
        for step in range(max_steps):
            actions = []
            for i, agent in enumerate(agents):
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                actions.append(action if 0 <= action < ACTIONS else 0)
            obs, terminated, truncated = env.step(actions)
            if terminated or truncated:
                break
        ranks = rank_players(env)
        if ranks[0] == 0:
            winners = [i for i in range(4) if ranks[i] == 0]
            if len(winners) == 1:
                wins += 1
    return float(wins) / num_matches

CURRICULUM_CONFIG = {
    0: {"max_steps": 100, "win_rate": 0.80},
    1: {"max_steps": 150, "win_rate": 0.70},
    2: {"max_steps": 200, "win_rate": 0.60},
    3: {"max_steps": 250, "win_rate": 0.50},
    4: {"max_steps": 300, "win_rate": 0.40},
    5: {"max_steps": 350, "win_rate": 0.40},
    6: {"max_steps": 400, "win_rate": 0.40},
    7: {"max_steps": 500, "win_rate": 0.35},
}

def train_ppo(model, device, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.ppo_lr, weight_decay=1e-4)
    snapshot_models = []
    
    initial_snap = PolicyValueNet().to(device)
    initial_snap.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    initial_snap.eval()
    snapshot_models.append(initial_snap)
    
    stage = 0
    eval_interval = 5
    
    for update in range(args.ppo_updates):
        stage_cfg = CURRICULUM_CONFIG.get(stage, {"max_steps": args.max_steps, "win_rate": 0.40})
        stage_max_steps = stage_cfg["max_steps"]
        stage_threshold = stage_cfg["win_rate"]
        
        # Evaluate and potentially advance stage
        if update > 0 and update % eval_interval == 0 and stage < 7:
            print(f"--- Evaluating Stage {stage} ---")
            win_rate = evaluate_agent_win_rate(model, device, stage, stage_max_steps, num_matches=10)
            print(f"Win Rate: {win_rate*100:.1f}% (Threshold: {stage_threshold*100:.1f}%)")
            if win_rate >= stage_threshold:
                stage += 1
                print(f"-> ADVANCED TO STAGE {stage}!")
            else:
                print("-> Retrying stage...")

        print(f"PPO Update {update + 1}/{args.ppo_updates} | Curriculum Stage {stage}")
        batch = collect_ppo_rollout(
            model,
            device=device,
            horizon=args.ppo_horizon,
            envs=args.ppo_envs_per_update,
            max_steps=stage_max_steps,
            seed=args.seed + 10000 * update,
            snapshot_models=snapshot_models,
            stage=stage,
        )
        if not batch.actions:
            print(f"PPO {update + 1}/{args.ppo_updates}: empty rollout")
            continue

        advantages, returns = compute_gae(batch.rewards, batch.dones, batch.values, args.gamma, args.gae_lambda)
        spatial = torch.from_numpy(np.asarray(batch.spatial, dtype=np.float32)).to(device)
        scalar = torch.from_numpy(np.asarray(batch.scalar, dtype=np.float32)).to(device)
        masks = torch.from_numpy(np.asarray(batch.masks, dtype=np.bool_)).to(device)
        actions = torch.tensor(batch.actions, dtype=torch.long, device=device)
        old_logprobs = torch.tensor(batch.logprobs, dtype=torch.float32, device=device)
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
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * dist.entropy().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
        print(f"PPO {update + 1}/{args.ppo_updates}: reward_mean={np.mean(batch.rewards):.3f}, samples={len(actions)}")
        
        # Save snapshot
        if (update + 1) % 10 == 0:
            snap = PolicyValueNet().to(device)
            snap.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
            snap.eval()
            snapshot_models.append(snap)


EXPORT_AGENT_TEMPLATE = """from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

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
    team_id = "HybridBCPPO_Shielded"
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0
        self.device = torch.device('cpu')
        self.model = PolicyValueNet()
        ckpt = torch.load(Path(__file__).with_name('model.pt'), map_location='cpu')
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        
        self.fallback_agent = None
        try:
            import importlib.util
            fallback_path = Path(__file__).parent / "7.py"
            if not fallback_path.exists():
                fallback_path = Path(__file__).parent / "4.py"
            if fallback_path.exists():
                spec = importlib.util.spec_from_file_location("fallback_teacher", fallback_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.fallback_agent = module.Agent(self.agent_id)
        except Exception:
            pass


    def act(self, obs):
        try:
            forced = _forced(obs, self.agent_id)
            if forced is not None:
                return int(forced)
                
            mask = _get_safe_actions_mask(obs, self.agent_id)
            spatial, scalar = _encode(obs, self.agent_id, self.step_count)
            
            with torch.inference_mode():
                s = torch.from_numpy(spatial).unsqueeze(0)
                a = torch.from_numpy(scalar).unsqueeze(0)
                logits, _ = self.model(s, a)
                probs = torch.softmax(logits, dim=-1)
                
                masked_log = logits.masked_fill(~torch.from_numpy(mask).unsqueeze(0).bool(), -1e9)
                action = int(torch.argmax(masked_log, dim=-1).item())
                
                prob = probs[0, action].item()
                if (prob < 0.25 or not mask[action]) and self.fallback_agent is not None:
                    try:
                        return int(self.fallback_agent.act(obs))
                    except Exception:
                        pass
            self.step_count += 1
            return action if 0 <= action <= 5 else 0
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
    parser.add_argument("--ppo-lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.015)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-checkpoint", default="checkpoints/hybrid_bc_ppo.pt")
    parser.add_argument("--export-dir", default="exports/hybrid_ppo_agent")
    parser.add_argument("--smoke-matches", type=int, default=2)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = PolicyValueNet().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if args.mode in {"bc", "full"}:
        dataset = collect_expert_dataset(args.bc_matches, args.max_steps, args.seed)
        train_behavior_cloning(model, dataset, device, args.bc_epochs, args.bc_batch_size, args.bc_lr, args.grad_clip)
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
