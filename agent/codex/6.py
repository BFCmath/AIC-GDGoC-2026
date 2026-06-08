from collections import deque


class Agent:
    """
    ApexHybridV6 for GDGoC Bomberland 2026.

    Design goal: general ladder strength, not anti-4.py overfit.
    - Exact-ish time-expanded survival for existing bombs and chain reactions.
    - Active farming because timeout tie-break is kills > boxes > items > bombs.
    - Target-search over all reachable safe states rather than nearest-target greed.
    - Bomb only when material/kill/trap value exists, or late safe tie-break pressure.
    """

    team_id = "ApexHybridV6"

    # Engine-compatible movement: docs names are wrong, these deltas are what the engine does.
    MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    CARDINALS = (1, 2, 3, 4)

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.bomb_radius_mem = {}     # (x, y, owner) -> radius at placement/sighting time
        self.last_pos = None
        self.stuck = 0
        self.visit = {}               # soft anti-loop map, decayed lazily
        self.last_action = 0

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
            my_radius = max(1, min(5, 1 + int(me[4])))

            if self.last_pos == my_pos:
                self.stuck += 1
            else:
                self.stuck = 0
            self.last_pos = my_pos
            self.visit[my_pos] = self.visit.get(my_pos, 0) + 1
            if self.turn % 40 == 0 and len(self.visit) > 70:
                self.visit = {p: max(1, v // 2) for p, v in self.visit.items() if v > 1}

            enemies = []
            for i, p in enumerate(players):
                if i != self.agent_id and int(p[2]) == 1:
                    enemies.append((i, (int(p[0]), int(p[1])), max(1, min(5, 1 + int(p[4]))), int(p[3])))

            danger = self._danger_schedule(grid, bombs, players, horizon=10)
            legal = self._legal_actions(grid, my_pos, bombs, bombs_left)

            # 1) Survival first: do not reason from static danger only; simulate time states.
            threatened = any(my_pos in danger.get(t, set()) for t in range(1, 8))
            if threatened:
                escape = self._survival_action(grid, my_pos, bombs, players, danger, enemies, horizon=10)
                if escape is not None:
                    self.last_action = escape
                    return escape
                immediate = [a for a in legal if a != 5 and self._next(my_pos, a) not in danger.get(1, set())]
                if immediate:
                    action = max(immediate, key=lambda a: self._fallback_score(grid, self._next(my_pos, a), bombs, enemies, danger))
                    self.last_action = action
                    return action
                self.last_action = 0
                return 0

            # 2) Free, immediate item pickup. Capacity is usually more valuable early.
            item_actions = []
            for a in legal:
                if a == 5:
                    continue
                p = self._next(my_pos, a)
                if not self._passable(grid, p[0], p[1]) or p in danger.get(1, set()):
                    continue
                cell = int(grid[p[0], p[1]])
                if cell in (3, 4):
                    item_actions.append((self._item_value(cell, bombs_left, my_radius) + 1.5 * self._mobility(grid, p, bombs)
                                         - 4.0 * self._contest_risk(p, enemies), a))
            if item_actions:
                action = max(item_actions)[1]
                self.last_action = action
                return action

            # 3) Bombing: value + verified escape. This is where most ladder points come from.
            if 5 in legal:
                bomb_value = self._bomb_value(grid, my_pos, my_radius, bombs, players, enemies, bombs_left)
                can_escape = False
                escape_space = 0
                if bomb_value >= 8 or (self.turn > 360 and enemies):
                    can_escape = self._can_escape_after_bomb(grid, my_pos, bombs, players, my_radius, enemies)
                    if can_escape:
                        escape_space = self._escape_space_after_bomb(grid, my_pos, bombs, players, my_radius)

                # Early/mid: require real material or tactical value. Late: safe activity matters for timeout.
                if can_escape:
                    boxes = self._count_boxes_in_blast(grid, my_pos, my_radius)
                    hit = len(self._enemies_in_blast(grid, my_pos, my_radius, enemies))
                    if ((bomb_value >= 18 and escape_space >= 1) or
                        (boxes >= 1 and bomb_value >= 11) or
                        (hit and bomb_value >= 10) or
                        (self.turn > 430 and escape_space >= 2 and bomb_value >= 7)):
                        self.last_action = 5
                        return 5

            # 4) Global short-horizon search over valuable safe states.
            action = self._best_action_by_search(grid, my_pos, bombs, players, danger, enemies, bombs_left, my_radius)
            if action is not None:
                self.last_action = action
                return action

            # 5) Safe mobile fallback; prefer novelty and avoid enemy bomb lines.
            candidates = [a for a in legal if a != 5 and self._next(my_pos, a) not in danger.get(1, set())]
            if not candidates:
                self.last_action = 0
                return 0
            if self.stuck >= 3:
                movers = [a for a in candidates if a != 0]
                if movers:
                    candidates = movers
            action = max(candidates, key=lambda a: self._fallback_score(grid, self._next(my_pos, a), bombs, enemies, danger))
            self.last_action = action
            return action

        except Exception:
            # In this ladder, one uncaught exception is worse than a mediocre move.
            return 0

    # ------------------------------------------------------------------
    # Core state model

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
        for k in list(self.bomb_radius_mem.keys()):
            if k not in seen:
                self.bomb_radius_mem.pop(k, None)

    def _radius_for_bomb(self, b, players):
        key = (int(b[0]), int(b[1]), int(b[3]))
        if key in self.bomb_radius_mem:
            return self.bomb_radius_mem[key]
        owner = int(b[3])
        return max(1, min(5, 1 + int(players[owner][4]))) if 0 <= owner < len(players) else 2

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
            bomb_list.append({"pos": pos, "timer": max(1, int(timer)), "radius": max(1, min(5, int(radius)))})

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

    def _danger_schedule(self, grid, bombs, players, horizon=10, extra_bomb=None):
        schedule = {t: set() for t in range(1, horizon + 1)}
        bomb_list, times = self._bomb_explosion_times(grid, bombs, players, extra_bomb)
        for b, t in zip(bomb_list, times):
            if 1 <= t <= horizon:
                schedule[t].update(self._blast_tiles(grid, b["pos"][0], b["pos"][1], b["radius"]))
        return schedule

    # ------------------------------------------------------------------
    # Geometry and legality

    def _next(self, pos, action):
        dx, dy = self.MOVES.get(action, (0, 0))
        return (pos[0] + dx, pos[1] + dy)

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and int(grid[x, y]) in (0, 3, 4)

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for r in range(1, int(radius) + 1):
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
        for a in self.CARDINALS:
            p = self._next(my_pos, a)
            if self._passable(grid, p[0], p[1]) and p not in bomb_positions:
                actions.append(a)
        if int(bombs_left) > 0 and my_pos not in bomb_positions:
            actions.append(5)
        return actions

    def _bomb_positions_alive_at(self, bombs, t, extra_bomb=None):
        # A bomb blocks movement until the step in which it explodes; its blast danger also blocks that step.
        out = {(int(b[0]), int(b[1])) for b in bombs if int(b[2]) > t}
        if extra_bomb is not None:
            pos, _, _, timer = extra_bomb
            if int(timer) > t:
                out.add(pos)
        return out

    # ------------------------------------------------------------------
    # Search and survival

    def _survival_action(self, grid, start, bombs, players, danger, enemies, horizon=10, extra_bomb=None, require_leave_blast=None):
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None
        best_score = -10**9

        while q:
            pos, t, first = q.popleft()
            future_bad = any(pos in danger.get(tt, set()) for tt in range(t + 1, horizon + 1))
            if t > 0 and not future_bad:
                score = 10.0 - 1.15 * t + 2.4 * self._mobility(grid, pos, bombs)
                score += 0.35 * self._reachable_space(grid, pos, bombs, limit=18)
                score -= 1.5 * self._enemy_bomb_threat(grid, pos, enemies)
                score -= 0.6 * self.visit.get(pos, 0)
                if require_leave_blast and pos not in require_leave_blast:
                    score += 15
                if score > best_score:
                    best_score = score
                    best = first
                    if t <= 3 and (not require_leave_blast or pos not in require_leave_blast):
                        return first

            if t >= horizon:
                continue
            blocked_next = self._bomb_positions_alive_at(bombs, t + 1, extra_bomb)
            for a in (0, 1, 2, 3, 4):
                npos = self._next(pos, a)
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                # Starting on a bomb is legal; stepping onto a bomb is not useful/safe.
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

    def _best_action_by_search(self, grid, start, bombs, players, danger, enemies, bombs_left, my_radius):
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None
        best_score = -10**9
        horizon = 13

        while q:
            pos, t, first = q.popleft()
            if t > 0:
                score = self._position_value(grid, pos, bombs, players, danger, enemies, bombs_left, my_radius, t)
                if score > best_score:
                    best_score = score
                    best = first

            if t >= horizon:
                continue
            blocked_next = self._bomb_positions_alive_at(bombs, t + 1)
            for a in (0, 1, 2, 3, 4):
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

        return best if best is not None and best_score > -12 else None

    # ------------------------------------------------------------------
    # Evaluation

    def _position_value(self, grid, pos, bombs, players, danger, enemies, bombs_left, my_radius, dist):
        cell = int(grid[pos[0], pos[1]])
        score = -1.65 * dist

        if cell in (3, 4):
            score += self._item_value(cell, bombs_left, my_radius)
            score -= 5.0 * self._contest_risk(pos, enemies)

        # Value of reaching a cell from which our next bomb is productive.
        if int(bombs_left) > 0 and pos not in {(int(b[0]), int(b[1])) for b in bombs}:
            boxes = self._count_boxes_in_blast(grid, pos, my_radius)
            if boxes:
                # Box count is timeout material and future item generation.
                score += 13.5 * boxes + 2.0 * min(4, self._mobility(grid, pos, bombs))
            kill_cell = self._kill_cell_value(grid, pos, my_radius, enemies, bombs)
            score += kill_cell

        score += 1.15 * self._mobility(grid, pos, bombs)
        score += 0.18 * self._reachable_space(grid, pos, bombs, limit=24)
        score -= 0.85 * self._enemy_bomb_threat(grid, pos, enemies)
        score -= 0.45 * self.visit.get(pos, 0)

        # Late-game engagement: center-adjacent cells create more box/enemy contact and help avoid passive draws.
        if self.turn > 160:
            score -= 0.10 * (abs(pos[0] - 6) + abs(pos[1] - 6))
        if self.stuck >= 3 and pos == self.last_pos:
            score -= 8

        if any(pos in danger.get(t, set()) for t in range(1, 5)):
            score -= 25
        return score

    def _item_value(self, cell, bombs_left, my_radius):
        if int(cell) == 4:  # capacity
            return 68 if int(bombs_left) <= 1 else 46
        if int(cell) == 3:  # radius
            return 62 if int(my_radius) <= 2 else 34
        return 0

    def _bomb_value(self, grid, pos, radius, bombs, players, enemies, bombs_left):
        boxes = self._count_boxes_in_blast(grid, pos, radius)
        hit = self._enemies_in_blast(grid, pos, radius, enemies)
        trap = self._trap_score(grid, pos, radius, enemies, bombs, players)
        chain = self._chain_pressure_value(grid, pos, radius, bombs, players)

        value = 12.5 * boxes + 25.0 * len(hit) + trap + chain
        # Bomb-count tie-break is last, so this is small unless late.
        if self.turn > 320 and enemies:
            value += 4.0
        if self.turn > 430:
            value += 6.0
        # Do not waste early bombs in empty corridors unless they strongly threaten a player.
        if boxes == 0 and not hit and trap < 10:
            value -= 10.0 if self.turn < 360 else 3.0
        # If we only have one bomb, no-box bombs slow down farming unless they are tactical.
        if int(bombs_left) <= 1 and boxes == 0 and trap < 16 and not hit:
            value -= 4.0
        return value

    def _trap_score(self, grid, pos, radius, enemies, bombs, players):
        if not enemies:
            return 0.0
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        extra = (pos, radius, self.agent_id, 7)
        danger = self._danger_schedule(grid, bombs, players, horizon=8, extra_bomb=extra)
        score = 0.0
        bp = {(int(b[0]), int(b[1])) for b in bombs}

        for _, epos, eradius, ebleft in enemies:
            dist = abs(epos[0] - pos[0]) + abs(epos[1] - pos[1])
            if epos in blast:
                escapes = self._escape_count_from(grid, epos, bombs, danger, horizon=7, extra_bomb=extra, max_count=5)
                local_exits = sum(1 for a in self.CARDINALS
                                  if self._passable(grid, *self._next(epos, a))
                                  and self._next(epos, a) not in blast
                                  and self._next(epos, a) not in bp)
                score += max(0.0, 32.0 - 6.0 * escapes - 3.0 * local_exits)
            elif dist <= radius + 2:
                # Near-miss pressure: useful for forcing movement and creating future farming/kill chances.
                score += max(0.0, 9.0 - 1.5 * dist)
            if ebleft <= 0 and dist <= radius + 3:
                score += 2.0
        return score

    def _kill_cell_value(self, grid, pos, radius, enemies, bombs):
        # How valuable it is to stand here and threaten a bomb next turn.
        if not enemies:
            return 0.0
        value = 0.0
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        bp = {(int(b[0]), int(b[1])) for b in bombs}
        for _, epos, _, ebleft in enemies:
            d = abs(epos[0] - pos[0]) + abs(epos[1] - pos[1])
            if epos in blast:
                exits = sum(1 for a in self.CARDINALS
                            if self._passable(grid, *self._next(epos, a))
                            and self._next(epos, a) not in blast
                            and self._next(epos, a) not in bp)
                value += max(8.0, 30.0 - 5.0 * exits - 1.5 * d)
            elif d <= radius + 2 and self.turn > 180:
                value += max(0.0, 10.0 - d)
        return value

    def _chain_pressure_value(self, grid, pos, radius, bombs, players):
        # Bonus for placing a bomb that can participate in useful chain reactions, but penalize immediate self-danger.
        if len(bombs) == 0:
            return 0.0
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        value = 0.0
        for b in bombs:
            bpos = (int(b[0]), int(b[1]))
            if bpos in blast:
                timer = int(b[2])
                br = self._radius_for_bomb(b, players)
                boxes = self._count_boxes_in_blast(grid, bpos, br)
                value += max(0.0, 4.0 - 0.4 * timer) + 2.5 * boxes
                if timer <= 2 and pos in self._blast_tiles(grid, bpos[0], bpos[1], br):
                    value -= 12.0
        return value

    def _can_escape_after_bomb(self, grid, pos, bombs, players, radius, enemies):
        extra = (pos, radius, self.agent_id, 7)
        danger = self._danger_schedule(grid, bombs, players, horizon=10, extra_bomb=extra)
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        return self._survival_action(grid, pos, bombs, players, danger, enemies, horizon=10,
                                     extra_bomb=extra, require_leave_blast=blast) is not None

    def _escape_space_after_bomb(self, grid, pos, bombs, players, radius):
        extra = (pos, radius, self.agent_id, 7)
        danger = self._danger_schedule(grid, bombs, players, horizon=10, extra_bomb=extra)
        blocked = self._bomb_positions_alive_at(bombs, 1, extra)
        count = 0
        for a in self.CARDINALS:
            p = self._next(pos, a)
            if self._passable(grid, p[0], p[1]) and p not in blocked and p not in danger.get(1, set()):
                count += 1
        return count

    def _escape_count_from(self, grid, start, bombs, danger, horizon=7, extra_bomb=None, max_count=6):
        q = deque([(start, 0)])
        seen = {(start, 0)}
        safe_count = 0
        while q:
            pos, t = q.popleft()
            if t > 0 and not any(pos in danger.get(tt, set()) for tt in range(t + 1, horizon + 1)):
                safe_count += 1
                if safe_count >= max_count:
                    return safe_count
            if t >= horizon:
                continue
            blocked = self._bomb_positions_alive_at(bombs, t + 1, extra_bomb)
            for a in (0, 1, 2, 3, 4):
                npos = self._next(pos, a)
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if npos in blocked and npos != start:
                    continue
                if npos in danger.get(t + 1, set()):
                    continue
                st = (npos, t + 1)
                if st in seen:
                    continue
                seen.add(st)
                q.append((npos, t + 1))
        return safe_count

    # ------------------------------------------------------------------
    # Small scoring helpers

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for x, y in self._blast_tiles(grid, pos[0], pos[1], radius) if int(grid[x, y]) == 2)

    def _enemies_in_blast(self, grid, pos, radius, enemies):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        return [eid for eid, epos, _, _ in enemies if epos in blast]

    def _mobility(self, grid, pos, bombs):
        bp = {(int(b[0]), int(b[1])) for b in bombs}
        count = 0
        for a in (0, 1, 2, 3, 4):
            p = self._next(pos, a)
            if self._passable(grid, p[0], p[1]) and (p not in bp or p == pos):
                count += 1
        return count

    def _reachable_space(self, grid, start, bombs, limit=24):
        bp = {(int(b[0]), int(b[1])) for b in bombs}
        q = deque([start])
        seen = {start}
        while q and len(seen) < limit:
            pos = q.popleft()
            for a in self.CARDINALS:
                p = self._next(pos, a)
                if p in seen or p in bp:
                    continue
                if not self._passable(grid, p[0], p[1]):
                    continue
                seen.add(p)
                q.append(p)
                if len(seen) >= limit:
                    break
        return len(seen)

    def _enemy_bomb_threat(self, grid, pos, enemies):
        penalty = 0.0
        for _, epos, radius, bombs_left in enemies:
            if bombs_left <= 0:
                continue
            if pos in self._blast_tiles(grid, epos[0], epos[1], radius):
                d = abs(pos[0] - epos[0]) + abs(pos[1] - epos[1])
                # Low mobility in enemy line is dangerous; high mobility can escape later.
                penalty += max(0.0, 7.0 - d)
            elif abs(pos[0] - epos[0]) + abs(pos[1] - epos[1]) <= 2:
                penalty += 1.5
        return penalty

    def _contest_risk(self, pos, enemies):
        # Items vanish if multiple players step onto them in the same tick.
        risk = 0.0
        for _, epos, _, _ in enemies:
            d = abs(pos[0] - epos[0]) + abs(pos[1] - epos[1])
            if d == 0:
                risk += 2.0
            elif d == 1:
                risk += 1.0
            elif d == 2:
                risk += 0.35
        return risk

    def _fallback_score(self, grid, pos, bombs, enemies, danger):
        score = 2.2 * self._mobility(grid, pos, bombs)
        score += 0.25 * self._reachable_space(grid, pos, bombs, limit=20)
        score -= 1.2 * self._enemy_bomb_threat(grid, pos, enemies)
        score -= 0.6 * self.visit.get(pos, 0)
        if self.turn > 180:
            score -= 0.12 * (abs(pos[0] - 6) + abs(pos[1] - 6))
        if any(pos in danger.get(t, set()) for t in range(1, 5)):
            score -= 30
        if self.stuck >= 3 and pos == self.last_pos:
            score -= 8
        return score
