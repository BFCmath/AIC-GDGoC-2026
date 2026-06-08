from collections import deque

class Agent:
    """
    Time-expanded rule/search hybrid for GDGoC Bomberland.
    Engine-compatible actions: 0=stop, 1=up, 2=down, 3=left, 4=right, 5=bomb.
    """
    team_id = "TimeSafeHybrid"

    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.bomb_radius_mem = {}  # (row, col, owner) -> radius at first sight
        self.last_pos = None
        self.stuck = 0

    def act(self, obs: dict) -> int:
        try:
            self.turn += 1
            grid = obs["map"]
            players = obs["players"]
            bombs = obs["bombs"]
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                return 0

            self._update_bomb_memory(bombs, players)

            me = players[self.agent_id]
            my_pos = (int(me[0]), int(me[1]))
            bombs_left = int(me[3])
            my_radius = 1 + int(me[4])

            if self.last_pos == my_pos:
                self.stuck += 1
            else:
                self.stuck = 0
            self.last_pos = my_pos

            enemies = [
                (i, (int(p[0]), int(p[1])), 1 + int(p[4]), int(p[3]))
                for i, p in enumerate(players)
                if i != self.agent_id and int(p[2]) == 1
            ]

            danger = self._danger_schedule(grid, bombs, players, horizon=9)
            legal = self._legal_actions(grid, my_pos, bombs, bombs_left)

            # 1) Survival first: exact time-expanded escape, not static “danger soon”.
            threatened = any(my_pos in danger.get(t, set()) for t in range(1, 8))
            if threatened:
                escape = self._safe_bfs_action(grid, my_pos, bombs, players, danger, horizon=9)
                if escape is not None:
                    return escape
                safe_now = [a for a in legal if a != 5 and self._next(my_pos, a) not in danger.get(1, set())]
                if safe_now:
                    return max(safe_now, key=lambda a: self._mobility(grid, self._next(my_pos, a), bombs))
                return 0

            # 2) Bomb only when it creates material/kill pressure and exact escape exists.
            if 5 in legal:
                boxes = self._count_boxes_in_blast(grid, my_pos, my_radius)
                hit_enemies = self._enemies_in_blast(grid, my_pos, my_radius, enemies)
                trap_score = self._trap_score(grid, my_pos, my_radius, enemies, bombs, players)
                value = boxes * 8 + len(hit_enemies) * 18 + trap_score
                if self.turn > 360 and boxes == 0 and enemies:
                    value += 4  # late pressure and bomb-count tie-breaker
                if value >= 8 and self._can_escape_after_bomb(grid, my_pos, bombs, players, my_radius):
                    if value >= 18 or self._escape_space_after_bomb(grid, my_pos, bombs, players, my_radius) >= 2:
                        return 5

            # 3) Otherwise pursue items, box-bomb cells, then enemies with time-safe BFS.
            targets = []
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    cell = int(grid[x, y])
                    if cell == 4:      # capacity
                        targets.append(((x, y), 45 if bombs_left <= 1 else 32))
                    elif cell == 3:    # radius
                        targets.append(((x, y), 38 if my_radius <= 2 else 24))

            for pos, count in self._box_spots(grid, bombs).items():
                targets.append((pos, 12 + count * 10))

            for _, epos, _, _ in enemies:
                dist = abs(epos[0] - my_pos[0]) + abs(epos[1] - my_pos[1])
                if my_radius >= 2 or self.turn > 220:
                    targets.append((epos, max(8, 28 - 2 * dist)))

            move = self._best_move_to_targets(grid, my_pos, bombs, players, danger, targets, enemies)
            if move is not None:
                return move

            # 4) Safe mobile fallback; soft-avoid cells an enemy could bomb next.
            candidates = [a for a in legal if a != 5 and self._next(my_pos, a) not in danger.get(1, set())]
            if not candidates:
                return 0
            if self.stuck >= 4:
                movers = [a for a in candidates if a != 0]
                if movers:
                    candidates = movers
            return max(
                candidates,
                key=lambda a: self._fallback_score(grid, self._next(my_pos, a), bombs, enemies, players, danger),
            )
        except Exception:
            return 0

    # ---------- state models ----------

    def _update_bomb_memory(self, bombs, players):
        seen = set()
        for b in bombs:
            x, y, owner = int(b[0]), int(b[1]), int(b[3])
            key = (x, y, owner)
            seen.add(key)
            if key not in self.bomb_radius_mem:
                r = 2
                if 0 <= owner < len(players):
                    r = 1 + int(players[owner][4])
                self.bomb_radius_mem[key] = max(1, min(5, r))
        for key in list(self.bomb_radius_mem.keys()):
            if key not in seen:
                self.bomb_radius_mem.pop(key, None)

    def _radius_for_bomb(self, b, players):
        key = (int(b[0]), int(b[1]), int(b[3]))
        if key in self.bomb_radius_mem:
            return self.bomb_radius_mem[key]
        owner = int(b[3])
        return 1 + int(players[owner][4]) if 0 <= owner < len(players) else 2

    def _bomb_explosion_times(self, grid, bombs, players, extra_bomb=None):
        bomb_list = []
        for b in bombs:
            bomb_list.append({
                "pos": (int(b[0]), int(b[1])),
                "timer": max(1, int(b[2])),
                "radius": self._radius_for_bomb(b, players),
            })
        if extra_bomb is not None:
            pos, radius, owner, timer = extra_bomb
            bomb_list.append({"pos": pos, "timer": max(1, int(timer)), "radius": radius})

        times = [b["timer"] for b in bomb_list]
        changed = True
        while changed:
            changed = False
            for i, b in enumerate(bomb_list):
                blast = self._blast_tiles(grid, b["pos"][0], b["pos"][1], b["radius"])
                ti = times[i]
                for j, other in enumerate(bomb_list):
                    if i != j and other["pos"] in blast and times[j] > ti:
                        times[j] = ti
                        changed = True
        return bomb_list, times

    def _danger_schedule(self, grid, bombs, players, horizon=9, extra_bomb=None):
        schedule = {t: set() for t in range(1, horizon + 1)}
        bomb_list, times = self._bomb_explosion_times(grid, bombs, players, extra_bomb)
        for b, t in zip(bomb_list, times):
            if 1 <= t <= horizon:
                schedule[t].update(self._blast_tiles(grid, b["pos"][0], b["pos"][1], b["radius"]))
        return schedule

    # ---------- geometry ----------

    def _next(self, pos, action):
        dx, dy = self.MOVES.get(action, (0, 0))
        return (pos[0] + dx, pos[1] + dy)

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and int(grid[x, y]) in (0, 3, 4)

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == 1:
                    break
                tiles.add((x, y))
                if cell == 2:
                    break
        return tiles

    def _legal_actions(self, grid, my_pos, bombs, bombs_left):
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        actions = [0]
        for a in [1, 2, 3, 4]:
            nx, ny = self._next(my_pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in bomb_positions:
                actions.append(a)
        if bombs_left > 0 and my_pos not in bomb_positions:
            actions.append(5)
        return actions

    def _bomb_positions_alive_at(self, bombs, t, extra_bomb=None):
        # Conservative and fast: ignore earlier chain removal for blocking. This only loses some routes; it does not create illegal moves.
        out = {(int(b[0]), int(b[1])) for b in bombs if int(b[2]) > t}
        if extra_bomb is not None:
            pos, _, _, timer = extra_bomb
            if int(timer) > t:
                out.add(pos)
        return out

    # ---------- planning ----------

    def _safe_bfs_action(self, grid, start, bombs, players, danger, horizon=9, extra_bomb=None, require_leave_blast=None):
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None
        best_score = -10**9
        while q:
            pos, t, first = q.popleft()
            future_bad = any(pos in danger.get(tt, set()) for tt in range(t + 1, horizon + 1))
            if t > 0 and not future_bad:
                score = -t + 2 * self._mobility(grid, pos, bombs)
                if require_leave_blast and pos not in require_leave_blast:
                    score += 10
                if score > best_score:
                    best_score = score
                    best = first
                    if t <= 3 and (not require_leave_blast or pos not in require_leave_blast):
                        return first
            if t >= horizon:
                continue
            blocked_next = self._bomb_positions_alive_at(bombs, t + 1, extra_bomb)
            for a in [0, 1, 2, 3, 4]:
                npos = self._next(pos, a)
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                # Starting on a just-placed bomb is legal; stepping onto a bomb is not.
                if npos in blocked_next and npos != start:
                    continue
                if npos in danger.get(t + 1, set()):
                    continue
                state = (npos, t + 1)
                if state in seen:
                    continue
                seen.add(state)
                q.append((npos, t + 1, a if first is None else first))
        return best

    def _best_move_to_targets(self, grid, start, bombs, players, danger, targets, enemies):
        if not targets:
            return None
        target_value = {}
        for pos, value in targets:
            if pos not in target_value or value > target_value[pos]:
                target_value[pos] = value

        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None
        best_score = -10**9
        while q:
            pos, t, first = q.popleft()
            if t > 0 and pos in target_value:
                score = target_value[pos] - 2.2 * t + 0.8 * self._mobility(grid, pos, bombs)
                score -= self._enemy_bomb_threat(grid, pos, enemies)
                if score > best_score:
                    best_score = score
                    best = first
            if t >= 10:
                continue
            blocked_next = self._bomb_positions_alive_at(bombs, t + 1)
            for a in [0, 1, 2, 3, 4]:
                npos = self._next(pos, a)
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if npos in blocked_next:
                    continue
                if npos in danger.get(t + 1, set()):
                    continue
                state = (npos, t + 1)
                if state in seen:
                    continue
                seen.add(state)
                q.append((npos, t + 1, a if first is None else first))
        return best if best is not None and best_score > -5 else None

    # ---------- scoring helpers ----------

    def _can_escape_after_bomb(self, grid, my_pos, bombs, players, radius):
        extra = (my_pos, radius, self.agent_id, 7)
        danger = self._danger_schedule(grid, bombs, players, horizon=9, extra_bomb=extra)
        blast = self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
        return self._safe_bfs_action(grid, my_pos, bombs, players, danger, horizon=9, extra_bomb=extra, require_leave_blast=blast) is not None

    def _escape_space_after_bomb(self, grid, my_pos, bombs, players, radius):
        extra = (my_pos, radius, self.agent_id, 7)
        danger = self._danger_schedule(grid, bombs, players, horizon=9, extra_bomb=extra)
        blocked = self._bomb_positions_alive_at(bombs, 1, extra)
        count = 0
        for a in [1, 2, 3, 4]:
            p = self._next(my_pos, a)
            if self._passable(grid, p[0], p[1]) and p not in blocked and p not in danger.get(1, set()):
                count += 1
        return count

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for x, y in self._blast_tiles(grid, pos[0], pos[1], radius) if int(grid[x, y]) == 2)

    def _enemies_in_blast(self, grid, pos, radius, enemies):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        return [eid for eid, epos, _, _ in enemies if epos in blast]

    def _trap_score(self, grid, pos, radius, enemies, bombs, players):
        if not enemies:
            return 0
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        score = 0
        for _, epos, _, _ in enemies:
            if epos in blast:
                score += max(0, 18 - 3 * self._mobility(grid, epos, bombs))
            dist = abs(epos[0] - pos[0]) + abs(epos[1] - pos[1])
            if dist <= radius + 1:
                score += max(0, 8 - dist)
        return score

    def _box_spots(self, grid, bombs):
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        spots = {}
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if int(grid[x, y]) != 2:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in bomb_positions:
                        spots[(nx, ny)] = spots.get((nx, ny), 0) + 1
        return spots

    def _mobility(self, grid, pos, bombs):
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        count = 0
        for a in [0, 1, 2, 3, 4]:
            p = self._next(pos, a)
            if self._passable(grid, p[0], p[1]) and (p not in bomb_positions or p == pos):
                count += 1
        return count

    def _enemy_bomb_threat(self, grid, pos, enemies):
        penalty = 0
        for _, epos, radius, bombs_left in enemies:
            if bombs_left <= 0:
                continue
            if pos in self._blast_tiles(grid, epos[0], epos[1], radius):
                d = abs(pos[0] - epos[0]) + abs(pos[1] - epos[1])
                penalty += max(0, 6 - d)
        return penalty

    def _fallback_score(self, grid, pos, bombs, enemies, players, danger):
        score = 2 * self._mobility(grid, pos, bombs)
        score -= self._enemy_bomb_threat(grid, pos, enemies)
        if self.turn > 180:
            score -= 0.15 * (abs(pos[0] - 6) + abs(pos[1] - 6))
        if any(pos in danger.get(t, set()) for t in range(1, 5)):
            score -= 20
        return score
