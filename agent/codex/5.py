"""
RiskCertifiedMetaBomber agent for GDGoC-HCMUS AI Challenge 2026 BomberGame.

Interface-compatible with the provided SimpleRuleAgent style:
    agent = RiskCertifiedMetaBomber(agent_id)
    action = agent.act(observation)

Design goals:
- Hard survival filter before any farming / attacking decision.
- Time-aware danger map with bomb chain reactions.
- Space-time BFS escape certification under 100ms on 13x13 maps.
- Conservative ladder-friendly style: farm safely, trap opportunistically,
  and reduce unnecessary risk late game.

Action space:
    0 STOP
    1 LEFT
    2 RIGHT
    3 UP
    4 DOWN
    5 PLACE_BOMB
"""

import random
from collections import deque, defaultdict


class RiskCertifiedMetaBomber:
    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    MOVE_ACTIONS = (0, 1, 2, 3, 4)
    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # Cells used by the starter kit according to the provided baseline:
    # 0 = grass/empty, 1 = wall, 2 = box, 3 = radius item, 4 = capacity item.
    PASSABLE_VALUES = {0, 3, 4}
    ITEM_VALUES = {3, 4}

    team_id = "RiskCertifiedMetaBomber"

    # Conservative horizon. Default bomb timer is 7, so 12 catches normal escape,
    # delayed chain effects, and short-term opponent pressure while staying cheap.
    HORIZON = 12
    BOMB_TIMER = 7
    MAX_RADIUS = 5

    def __init__(self, agent_id):
        self.agent_id = int(agent_id)
        self.rng = random.Random(911 + self.agent_id)
        self.step_counter = 0
        self.last_action = 0
        self.my_history = deque(maxlen=16)
        self.enemy_history = defaultdict(lambda: deque(maxlen=24))
        self.prev_grid_box_count = None
        self.estimated_boxes_destroyed_global = 0
        self.estimated_bombs_placed = 0

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    def act(self, observation):
        self.step_counter += 1

        try:
            grid = observation["map"]
            players = observation["players"]
            bombs = observation.get("bombs", [])
        except Exception:
            # If observation is malformed, prefer not to crash the runner.
            self.last_action = 0
            return 0

        if self.agent_id >= len(players):
            self.last_action = 0
            return 0

        me = players[self.agent_id]
        if len(me) < 3 or int(me[2]) != 1:
            self.last_action = 0
            return 0

        my_x, my_y = int(me[0]), int(me[1])
        my_pos = (my_x, my_y)
        bombs_left = int(me[3]) if len(me) > 3 else 0
        radius_bonus = int(me[4]) if len(me) > 4 else 0
        my_radius = max(1, min(self.MAX_RADIUS, radius_bonus + 1))
        step = self._get_step(observation)

        self._update_memory(grid, players, my_pos)

        enemies = self._alive_enemies(players)
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs if len(b) >= 2}

        # Movement is blocked by bombs, walls, and boxes. Opponents are deliberately
        # not hard blockers because the rules allow multiple agents on one cell.
        hard_blocked = set(bomb_positions)
        hard_blocked.discard(my_pos)  # allow stepping away from a bomb under us

        base_bomb_infos = self._parse_bombs(grid, bombs, players)
        base_chain_times = self._chain_reaction_times(grid, base_bomb_infos)
        base_danger = self._build_danger_by_time(grid, base_bomb_infos, base_chain_times, self.HORIZON)
        base_blocks = self._build_bomb_blocks_by_time(base_bomb_infos, base_chain_times, self.HORIZON)
        soft_opp_blast = self._opponent_bomb_threat(grid, players, enemies)

        current_danger_time = self._earliest_danger_time(my_pos, base_danger)
        emergency = current_danger_time is not None and current_danger_time <= 3

        candidates = self._candidate_actions(grid, my_pos, hard_blocked, bombs_left, bomb_positions)
        if not candidates:
            self.last_action = 0
            return 0

        scored = []
        unsafe_fallback = []

        for action in candidates:
            next_pos = self._next_pos(my_pos, action) if action in self.MOVES else my_pos
            extra_bomb = None
            if action == 5:
                extra_bomb = {
                    "x": my_pos[0],
                    "y": my_pos[1],
                    "timer": self.BOMB_TIMER,
                    "owner": self.agent_id,
                    "radius": my_radius,
                    "hypothetical": True,
                }

            bomb_infos = base_bomb_infos if extra_bomb is None else base_bomb_infos + [extra_bomb]
            chain_times = base_chain_times if extra_bomb is None else self._chain_reaction_times(grid, bomb_infos)
            danger = base_danger if extra_bomb is None else self._build_danger_by_time(
                grid, bomb_infos, chain_times, self.HORIZON
            )
            blocks = base_blocks if extra_bomb is None else self._build_bomb_blocks_by_time(
                bomb_infos, chain_times, self.HORIZON
            )

            safety = self._safety_metrics(grid, next_pos, danger, blocks, self.HORIZON)
            fallback_score = self._fallback_survival_score(next_pos, safety, danger, soft_opp_blast)
            if action == 5:
                # In a no-certified-escape situation, adding a new bomb is almost
                # never the least-bad survival action. Keep it as a last resort only.
                fallback_score -= 1000.0
            unsafe_fallback.append((fallback_score, action))

            if not safety["survivable"]:
                continue

            # Placing a bomb with no value creates self-risk and wastes tempo.
            if action == 5:
                bomb_value = self._bomb_value(
                    grid=grid,
                    my_pos=my_pos,
                    my_radius=my_radius,
                    enemies=enemies,
                    danger=danger,
                    blocks=blocks,
                    players=players,
                    step=step,
                )
                if bomb_value < self._bomb_threshold(step, emergency):
                    continue
            else:
                bomb_value = 0.0

            score = self._score_action(
                grid=grid,
                players=players,
                my_pos=my_pos,
                next_pos=next_pos,
                action=action,
                safety=safety,
                base_danger=base_danger,
                danger=danger,
                blocks=blocks,
                hard_blocked=hard_blocked,
                soft_opp_blast=soft_opp_blast,
                enemies=enemies,
                bombs_left=bombs_left,
                my_radius=my_radius,
                bomb_value=bomb_value,
                step=step,
                emergency=emergency,
            )
            scored.append((score, action))

        if scored:
            scored.sort(reverse=True, key=lambda x: x[0])
            chosen = scored[0][1]
        else:
            # No certified action exists. Pick the least-bad move instead of crashing
            # or random-walking into an immediate blast.
            unsafe_fallback.sort(reverse=True, key=lambda x: x[0])
            chosen = unsafe_fallback[0][1] if unsafe_fallback else 0

        if chosen == 5:
            self.estimated_bombs_placed += 1
        self.last_action = chosen
        return int(chosen)

    # ---------------------------------------------------------------------
    # Core scoring
    # ---------------------------------------------------------------------
    def _score_action(
        self,
        grid,
        players,
        my_pos,
        next_pos,
        action,
        safety,
        base_danger,
        danger,
        blocks,
        hard_blocked,
        soft_opp_blast,
        enemies,
        bombs_left,
        my_radius,
        bomb_value,
        step,
        emergency,
    ):
        # Survival dominates everything. The constants are intentionally large:
        # ladder rating punishes early death more than it rewards flashy attacks.
        score = 10000.0
        score += 120.0 * safety["max_depth"]
        score += 10.0 * safety["terminal_count"]
        score += 2.0 * safety["unique_safe_cells"]
        score += 4.0 * safety["early_safe_cells"]

        # Immediate danger mode: do not get distracted by items or boxes.
        if emergency:
            earliest = self._earliest_danger_time(next_pos, danger)
            if earliest is not None:
                score -= 300.0 / max(1, earliest)
            score += 35.0 * self._distance_from_bombs(next_pos, blocks)
            if action == 0:
                score -= 50.0
            return score + self.rng.random() * 0.01

        cell = self._cell(grid, next_pos[0], next_pos[1])

        # Item pickup. Capacity is very valuable early because it accelerates box
        # farming; radius becomes stronger once there are corridors and opponents.
        if cell == 4:
            score += 95.0 if bombs_left <= 1 else 55.0
        elif cell == 3:
            score += 80.0 if my_radius <= 2 else 42.0

        # Soft distance-to-item reward, but only through safe-ish routes.
        item_targets = self._item_tiles(grid, prefer_capacity=(bombs_left <= 1), prefer_radius=(my_radius <= 2))
        if item_targets:
            d_item = self._distance_to_targets(grid, next_pos, hard_blocked, item_targets, danger, max_depth=9)
            if d_item is not None:
                score += max(0.0, 44.0 - 5.0 * d_item)

        # Bomb value includes boxes, potential kills, and trap pressure.
        if action == 5:
            score += bomb_value
            # Reduce low-value bomb spam, especially late when survival/tie-break
            # conservation matters.
            if step >= 350 and bomb_value < 85.0:
                score -= 70.0
        else:
            # Move toward good bombing spots when not currently bombing.
            spots = self._valuable_bomb_spots(grid, hard_blocked, my_radius)
            if spots:
                d_spot = self._distance_to_targets(grid, next_pos, hard_blocked, spots, base_danger, max_depth=10)
                if d_spot is not None:
                    score += max(0.0, 36.0 - 3.5 * d_spot)

        # Opponent pressure: prefer positions that approach weak/trappable enemies,
        # but avoid hugging bomb-capable enemies without a reason.
        if enemies:
            nearest_enemy_dist = min(self._manhattan(next_pos, epos) for _, epos, _ in enemies)
            if nearest_enemy_dist <= 2 and bombs_left > 0 and safety["unique_safe_cells"] >= 8:
                score += 12.0
            if nearest_enemy_dist <= 1 and next_pos in soft_opp_blast:
                score -= 38.0

        # Potential opponent bomb cells are not hard-forbidden, but they are costly.
        if next_pos in soft_opp_blast:
            score -= 32.0

        # Time-specific danger penalties. Passing through future blast lines can be
        # legal, but standing on cells that explode soon should be avoided.
        for t in range(1, min(5, len(danger))):
            if next_pos in danger[t]:
                score -= 160.0 / t

        # Mobility and anti-dead-end shaping.
        mobility = self._local_mobility(grid, next_pos, hard_blocked)
        score += 6.0 * mobility
        if mobility <= 1:
            score -= 28.0

        # Avoid pointless oscillation unless escaping or collecting an item.
        if action != 5 and cell not in self.ITEM_VALUES and next_pos in self.my_history:
            score -= 4.5

        # STOP is sometimes correct but should not dominate in open safe space.
        if action == 0 and not emergency:
            score -= 5.0

        # Late-game ladder behavior: if still alive near the end, value robust
        # survival more and demand higher confidence for aggression.
        if step >= 400:
            score += 4.0 * safety["terminal_count"]
            if action == 5 and bomb_value < 120.0:
                score -= 35.0

        return score + self.rng.random() * 0.01

    def _bomb_value(self, grid, my_pos, my_radius, enemies, danger, blocks, players, step):
        blast = self._blast_tiles(grid, my_pos[0], my_pos[1], my_radius)
        boxes = sum(1 for p in blast if self._cell(grid, p[0], p[1]) == 2)

        value = 0.0
        value += 18.0 * boxes

        # Adjacent/corridor enemies often die to simple bombs, but we still estimate
        # their escape volume using the same certified planner.
        for enemy_id, enemy_pos, enemy_radius in enemies:
            dist = self._manhattan(my_pos, enemy_pos)
            enemy_in_blast_line = enemy_pos in blast
            enemy_safety = self._safety_metrics(grid, enemy_pos, danger, blocks, min(self.HORIZON, 10))
            enemy_escape = enemy_safety["unique_safe_cells"]
            enemy_terminal = enemy_safety["terminal_count"]

            if enemy_in_blast_line:
                value += 42.0
                if enemy_escape <= 4:
                    value += 88.0
                elif enemy_escape <= 8:
                    value += 40.0
            elif dist <= my_radius + 2 and enemy_escape <= 5:
                value += 28.0

            if dist == 1:
                value += 22.0
            if enemy_terminal == 0:
                value += 55.0

            # If the enemy is likely random/greedy, direct traps are more valuable.
            profile = self._classify_enemy(enemy_id)
            if profile in ("random", "greedy") and dist <= 3:
                value += 18.0

        # Control value: bombs that open boxes near our current region are useful.
        if boxes >= 2:
            value += 20.0

        # Do not over-bomb late unless it creates real kill or box value.
        if step >= 350 and boxes == 0:
            value -= 25.0

        return value

    def _bomb_threshold(self, step, emergency):
        if emergency:
            return 10**9  # never voluntarily bomb while escaping
        if step >= 400:
            return 70.0
        if step >= 300:
            return 52.0
        return 28.0

    def _fallback_survival_score(self, pos, safety, danger, soft_opp_blast):
        score = 0.0
        score += 100.0 * safety.get("max_depth", 0)
        score += 8.0 * safety.get("unique_safe_cells", 0)
        earliest = self._earliest_danger_time(pos, danger)
        if earliest is None:
            score += 500.0
        else:
            score += 30.0 * earliest
        if pos in soft_opp_blast:
            score -= 30.0
        return score

    # ---------------------------------------------------------------------
    # Safety kernel: time-aware bombs, danger, chain reactions, escape BFS
    # ---------------------------------------------------------------------
    def _parse_bombs(self, grid, bombs, players):
        infos = []
        for b in bombs:
            if len(b) < 3:
                continue
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner = int(b[3]) if len(b) > 3 else -1
            radius = 2
            if 0 <= owner < len(players) and len(players[owner]) > 4:
                radius = max(1, min(self.MAX_RADIUS, int(players[owner][4]) + 1))
            if self._in_bounds(grid, bx, by):
                infos.append({
                    "x": bx,
                    "y": by,
                    "timer": max(1, timer),
                    "owner": owner,
                    "radius": radius,
                    "hypothetical": False,
                })
        return infos

    def _chain_reaction_times(self, grid, bomb_infos):
        n = len(bomb_infos)
        times = [max(1, int(b["timer"])) for b in bomb_infos]
        if n <= 1:
            return times

        # Fixed-point relaxation: if bomb i explodes before bomb j and its blast
        # reaches j, j inherits i's explosion time.
        changed = True
        loops = 0
        while changed and loops < n + 2:
            changed = False
            loops += 1
            for i, bi in enumerate(bomb_infos):
                blast_i = self._blast_tiles(grid, bi["x"], bi["y"], bi["radius"])
                ti = times[i]
                for j, bj in enumerate(bomb_infos):
                    if i == j:
                        continue
                    if times[j] > ti and (bj["x"], bj["y"]) in blast_i:
                        times[j] = ti
                        changed = True
        return times

    def _build_danger_by_time(self, grid, bomb_infos, chain_times, horizon):
        danger = [set() for _ in range(horizon + 1)]
        for b, t in zip(bomb_infos, chain_times):
            if 0 <= t <= horizon:
                danger[t].update(self._blast_tiles(grid, b["x"], b["y"], b["radius"]))
        return danger

    def _build_bomb_blocks_by_time(self, bomb_infos, chain_times, horizon):
        blocks = [set() for _ in range(horizon + 1)]
        for b, explode_t in zip(bomb_infos, chain_times):
            bx, by = b["x"], b["y"]
            until = min(horizon, max(1, explode_t))
            for t in range(1, until + 1):
                blocks[t].add((bx, by))
        return blocks

    def _safety_metrics(self, grid, start, danger, blocks, horizon):
        # Start represents our position after the chosen action has been applied.
        if not self._passable(grid, start[0], start[1]):
            return self._empty_safety()
        if len(danger) > 1 and start in danger[1]:
            return self._empty_safety()

        origin_privilege = start in blocks[1] if len(blocks) > 1 else False
        q = deque([(start, 1, origin_privilege)])
        seen = {(start, 1, origin_privilege)}

        max_depth = 1
        terminal_count = 0
        unique_safe = {start}
        early_safe = set()

        while q:
            pos, t, privilege = q.popleft()
            max_depth = max(max_depth, t)

            if t <= 4:
                early_safe.add(pos)
            if t >= horizon:
                terminal_count += 1
                continue

            nt = t + 1
            for action in self.MOVE_ACTIONS:
                npos = self._next_pos(pos, action)
                nx, ny = npos
                if not self._passable(grid, nx, ny):
                    continue
                if nt < len(danger) and npos in danger[nt]:
                    continue

                blocked_by_bomb = nt < len(blocks) and npos in blocks[nt]
                if blocked_by_bomb:
                    # Only allow staying on the origin bomb while we have not left it.
                    if not (privilege and pos == start and npos == start):
                        continue

                new_privilege = bool(privilege and npos == start)
                key = (npos, nt, new_privilege)
                if key in seen:
                    continue
                seen.add(key)
                unique_safe.add(npos)
                q.append((npos, nt, new_privilege))

        # Certified if we can survive until the planning horizon. With all known
        # bomb timers <= 7 normally, reaching horizon means the immediate threat is solved.
        survivable = terminal_count > 0
        return {
            "survivable": survivable,
            "max_depth": max_depth,
            "terminal_count": terminal_count,
            "unique_safe_cells": len(unique_safe),
            "early_safe_cells": len(early_safe),
        }

    def _empty_safety(self):
        return {
            "survivable": False,
            "max_depth": 0,
            "terminal_count": 0,
            "unique_safe_cells": 0,
            "early_safe_cells": 0,
        }

    def _earliest_danger_time(self, pos, danger):
        for t in range(1, len(danger)):
            if pos in danger[t]:
                return t
        return None

    def _opponent_bomb_threat(self, grid, players, enemies):
        # Soft threat model: any living opponent with bombs may place a bomb on its
        # current cell. This catches simple chase/bait traps without over-constraining.
        threat = set()
        for enemy_id, epos, _ in enemies:
            p = players[enemy_id]
            bombs_left = int(p[3]) if len(p) > 3 else 0
            radius = max(1, min(self.MAX_RADIUS, int(p[4]) + 1)) if len(p) > 4 else 2
            if bombs_left > 0:
                threat.update(self._blast_tiles(grid, epos[0], epos[1], radius))
        return threat

    # ---------------------------------------------------------------------
    # Movement, targeting, and features
    # ---------------------------------------------------------------------
    def _candidate_actions(self, grid, my_pos, hard_blocked, bombs_left, bomb_positions):
        actions = []
        for action in self.MOVE_ACTIONS:
            nx, ny = self._next_pos(my_pos, action)
            if not self._passable(grid, nx, ny):
                continue
            if (nx, ny) in hard_blocked:
                continue
            actions.append(action)

        if bombs_left > 0 and my_pos not in bomb_positions:
            actions.append(5)
        return actions

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES.get(action, (0, 0))
        return pos[0] + dx, pos[1] + dy

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in self.DIRS:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = self._cell(grid, x, y)
                if cell == 1:
                    break
                tiles.add((x, y))
                if cell == 2:
                    break
        return tiles

    def _item_tiles(self, grid, prefer_capacity=False, prefer_radius=False):
        preferred = set()
        if prefer_radius:
            preferred.add(3)
        if prefer_capacity:
            preferred.add(4)

        preferred_tiles = set()
        all_tiles = set()
        for x in range(self._width(grid)):
            for y in range(self._height(grid)):
                val = self._cell(grid, x, y)
                if val in self.ITEM_VALUES:
                    all_tiles.add((x, y))
                    if val in preferred:
                        preferred_tiles.add((x, y))
        return preferred_tiles if preferred_tiles else all_tiles

    def _valuable_bomb_spots(self, grid, hard_blocked, radius):
        spots = set()
        for x in range(self._width(grid)):
            for y in range(self._height(grid)):
                if not self._passable(grid, x, y) or (x, y) in hard_blocked:
                    continue
                boxes = self._count_boxes_in_blast(grid, (x, y), radius)
                if boxes > 0:
                    spots.add((x, y))
        return spots

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for x, y in self._blast_tiles(grid, pos[0], pos[1], radius) if self._cell(grid, x, y) == 2)

    def _distance_to_targets(self, grid, start, hard_blocked, targets, danger, max_depth=10):
        if start in targets:
            return 0
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if d >= max_depth:
                continue
            nd = d + 1
            for action in self.MOVE_ACTIONS:
                npos = self._next_pos(pos, action)
                nx, ny = npos
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in hard_blocked:
                    continue
                # Avoid paths that step into imminent known explosions.
                if nd < len(danger) and npos in danger[nd]:
                    continue
                if nd + 1 < len(danger) and npos in danger[nd + 1]:
                    continue
                if npos in targets:
                    return nd
                seen.add(npos)
                q.append((npos, nd))
        return None

    def _local_mobility(self, grid, pos, hard_blocked):
        count = 0
        for action in self.MOVE_ACTIONS:
            npos = self._next_pos(pos, action)
            if self._passable(grid, npos[0], npos[1]) and npos not in hard_blocked:
                count += 1
        return count

    def _distance_from_bombs(self, pos, blocks):
        bomb_cells = set()
        for s in blocks[: min(5, len(blocks))]:
            bomb_cells.update(s)
        if not bomb_cells:
            return 5
        return min(5, min(self._manhattan(pos, b) for b in bomb_cells))

    # ---------------------------------------------------------------------
    # Opponent memory and simple profile classifier
    # ---------------------------------------------------------------------
    def _update_memory(self, grid, players, my_pos):
        self.my_history.append(my_pos)
        for i, p in enumerate(players):
            if i == self.agent_id or len(p) < 3 or int(p[2]) != 1:
                continue
            self.enemy_history[i].append((int(p[0]), int(p[1])))

        box_count = self._box_count(grid)
        if self.prev_grid_box_count is not None and box_count < self.prev_grid_box_count:
            self.estimated_boxes_destroyed_global += self.prev_grid_box_count - box_count
        self.prev_grid_box_count = box_count

    def _alive_enemies(self, players):
        enemies = []
        for i, p in enumerate(players):
            if i == self.agent_id or len(p) < 3 or int(p[2]) != 1:
                continue
            epos = (int(p[0]), int(p[1]))
            radius = max(1, min(self.MAX_RADIUS, int(p[4]) + 1)) if len(p) > 4 else 2
            enemies.append((i, epos, radius))
        return enemies

    def _classify_enemy(self, enemy_id):
        hist = self.enemy_history.get(enemy_id)
        if not hist or len(hist) < 8:
            return "unknown"

        # Very lightweight behavior hints. This is intentionally cheap and robust.
        unique_positions = len(set(hist))
        total = len(hist)
        if unique_positions <= max(2, total // 5):
            return "random"  # often stuck/oscillating; easy to trap or ignore

        # Greedy/cautious labels are rough. The scorer only uses them as tiny biases.
        recent = list(hist)[-6:]
        backtracks = 0
        for a, b, c in zip(recent, recent[1:], recent[2:]):
            if a == c and a != b:
                backtracks += 1
        if backtracks >= 2:
            return "cautious"
        if unique_positions >= total * 0.75:
            return "greedy"
        return "unknown"

    # ---------------------------------------------------------------------
    # Grid helpers
    # ---------------------------------------------------------------------
    def _get_step(self, observation):
        for key in ("step", "tick", "turn", "timestep", "time"):
            if key in observation:
                try:
                    return int(observation[key])
                except Exception:
                    pass
        return self.step_counter

    def _width(self, grid):
        if hasattr(grid, "shape"):
            return int(grid.shape[0])
        return len(grid)

    def _height(self, grid):
        if hasattr(grid, "shape"):
            return int(grid.shape[1])
        return len(grid[0]) if grid else 0

    def _cell(self, grid, x, y):
        try:
            return int(grid[x, y])
        except Exception:
            return int(grid[x][y])

    def _in_bounds(self, grid, x, y):
        return 0 <= x < self._width(grid) and 0 <= y < self._height(grid)

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and self._cell(grid, x, y) in self.PASSABLE_VALUES

    def _box_count(self, grid):
        cnt = 0
        for x in range(self._width(grid)):
            for y in range(self._height(grid)):
                if self._cell(grid, x, y) == 2:
                    cnt += 1
        return cnt

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Convenience aliases. Many starter kits import Agent from agent.py; others may
# ask teams to expose a class similar to the sample. These aliases make both easy.
Agent = RiskCertifiedMetaBomber
SimpleRuleAgent = RiskCertifiedMetaBomber
