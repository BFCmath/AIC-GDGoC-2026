from collections import deque

class Agent:
    """
    Assassin-aware time-expanded hybrid for GDGoC Bomberland.
    Engine-compatible actions: 0=stop, 1=up, 2=down, 3=left, 4=right, 5=bomb.
    """
    team_id = "SeatAwareAssassinV20"

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

            # Anti-trap layer: top ladder agents tend to punish cells that are in their
            # current bomb line. Leave those low-mobility lines before an actual bomb exists.
            if self._enemy_future_bomb_risk(grid, my_pos, enemies, bombs) >= 14:
                anti = [a for a in legal if a != 5 and self._next(my_pos, a) not in danger.get(1, set())]
                if anti:
                    best_anti = max(anti, key=lambda a: self._fallback_score(grid, self._next(my_pos, a), bombs, enemies, players, danger))
                    if self._enemy_future_bomb_risk(grid, self._next(my_pos, best_anti), enemies, bombs) < self._enemy_future_bomb_risk(grid, my_pos, enemies, bombs):
                        return best_anti

            current_kill_score = 0
            if self.agent_id >= 2 and 5 in legal and enemies and (my_radius >= 2 or self.turn > 160):
                current_kill_score = self._candidate_kill_value(grid, my_pos, my_radius, enemies, bombs, players, strict=True)
                if current_kill_score >= 115 and self._can_escape_after_bomb(grid, my_pos, bombs, players, my_radius):
                    return 5

            # Immediate item pickup before bombing: items are the 3rd timeout tie-breaker
            # and capacity/radius unlock more future farming and trap pressure.
            immediate_items = []
            for a in legal:
                if a == 5:
                    continue
                p = self._next(my_pos, a)
                if self._passable(grid, p[0], p[1]) and int(grid[p[0], p[1]]) in (3, 4) and p not in danger.get(1, set()):
                    immediate_items.append(a)
            if immediate_items:
                return max(immediate_items, key=lambda a: (int(grid[self._next(my_pos, a)[0], self._next(my_pos, a)[1]]) == 4, self._mobility(grid, self._next(my_pos, a), bombs)))

            # 2) Bomb only when it creates material/kill pressure and exact escape exists.
            if 5 in legal:
                boxes = self._count_boxes_in_blast(grid, my_pos, my_radius)
                hit_enemies = self._enemies_in_blast(grid, my_pos, my_radius, enemies)
                trap_score = self._trap_score(grid, my_pos, my_radius, enemies, bombs, players)
                value = boxes * 12 + len(hit_enemies) * 24 + trap_score
                if current_kill_score >= 70:
                    value += min(45, current_kill_score - 55)
                if self.turn > 330 and boxes == 0 and enemies:
                    value += 8  # late pressure and bomb-count tie-breaker
                # More aggressive than pure survival: timeout ranking rewards boxes/items/bombs after survival.
                if value >= 10 and self._can_escape_after_bomb(grid, my_pos, bombs, players, my_radius):
                    escape_space = self._escape_space_after_bomb(grid, my_pos, bombs, players, my_radius)
                    if boxes >= 1 or hit_enemies or trap_score >= 4 or current_kill_score >= 85 or escape_space >= 1:
                        return 5

            # 3) Otherwise pursue items, box-bomb cells, then enemies with time-safe BFS.
            targets = []
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    cell = int(grid[x, y])
                    if cell == 4:      # capacity
                        targets.append(((x, y), 58 if bombs_left <= 1 else 38))
                    elif cell == 3:    # radius
                        targets.append(((x, y), 50 if my_radius <= 2 else 28))

            for pos, count in self._box_spots(grid, bombs).items():
                boxes_at = self._count_boxes_in_blast(grid, pos, my_radius)
                targets.append((pos, 16 + boxes_at * 15 + count * 3))

            # Late/powered assassination mode: after we have enough board control,
            # seek only high-confidence trap cells. Early game remains V9-style farming.
            powered = (my_radius >= 3 and bombs_left >= 2) or self.turn > 280 or len(enemies) <= 1
            if self.agent_id >= 2 and enemies and powered:
                targets.extend(self._assassin_bomb_targets(grid, my_pos, my_radius, enemies, bombs, players, danger))

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
                score = target_value[pos] - 1.85 * t + 0.7 * self._mobility(grid, pos, bombs)
                score -= 0.65 * self._enemy_bomb_threat(grid, pos, enemies)
                score -= 0.20 * self._enemy_future_bomb_risk(grid, pos, enemies, bombs)
                if score > best_score:
                    best_score = score
                    best = first
            if t >= 12:
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
        return best if best is not None and best_score > -8 else None

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

    def _extra_bomb_time(self, grid, bombs, players, extra_bomb):
        bomb_list, times = self._bomb_explosion_times(grid, bombs, players, extra_bomb)
        if not times:
            return 7
        return max(1, min(11, int(times[-1])))

    def _survival_state_count(self, grid, start, bombs, danger, horizon=8, extra_bomb=None, cap=24):
        if not self._passable(grid, start[0], start[1]):
            return 0
        current = {start}
        for t in range(1, horizon + 1):
            blocked = self._bomb_positions_alive_at(bombs, t, extra_bomb)
            nxt = set()
            for pos in current:
                for a in [0, 1, 2, 3, 4]:
                    npos = self._next(pos, a)
                    if not self._passable(grid, npos[0], npos[1]):
                        continue
                    # If the state starts on a newly placed bomb, stopping there is legal in the engine;
                    # moving onto any other bomb is not.
                    if npos in blocked and npos != start:
                        continue
                    if npos in danger.get(t, set()):
                        continue
                    nxt.add(npos)
                    if len(nxt) >= cap:
                        return cap
            if not nxt:
                return 0
            current = nxt
        return len(current)

    def _enemy_next_positions(self, grid, pos, bombs, danger=None):
        # Opponent gets one simultaneous action before our newly placed bomb exists.
        # Use this to avoid over-valuing bombs that only hit an idle enemy.
        blocked = {(int(b[0]), int(b[1])) for b in bombs}
        out = []
        for a in [0, 1, 2, 3, 4]:
            p = self._next(pos, a)
            if not self._passable(grid, p[0], p[1]):
                continue
            if p in blocked and p != pos:
                continue
            if danger is not None and p in danger.get(1, set()):
                continue
            out.append(p)
        return out or [pos]

    def _line_attack_cells(self, grid, target_pos, radius):
        cells = []
        tx, ty = target_pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(0, radius + 1):
                x, y = tx - dx * r, ty - dy * r
                if not self._in_bounds(grid, x, y):
                    break
                if not self._passable(grid, x, y):
                    # The target may stand on passable only; walls/boxes cannot host our bomb.
                    if (x, y) != target_pos:
                        break
                    continue
                if target_pos in self._blast_tiles(grid, x, y, radius):
                    cells.append((x, y))
        # Deduplicate while preserving small-list order.
        seen = set(); out = []
        for c in cells:
            if c not in seen:
                seen.add(c); out.append(c)
        return out

    def _candidate_kill_value(self, grid, bomb_pos, radius, enemies, bombs, players, strict=False):
        if not self._passable(grid, bomb_pos[0], bomb_pos[1]):
            return 0
        if bomb_pos in {(int(b[0]), int(b[1])) for b in bombs}:
            return 0

        # Direct model: if the enemy starts from the observed state and plays evasively.
        extra7 = (bomb_pos, radius, self.agent_id, 7)
        t7 = self._extra_bomb_time(grid, bombs, players, extra7)
        horizon7 = max(7, min(11, t7 + 1))
        danger_extra7 = self._danger_schedule(grid, bombs, players, horizon=horizon7, extra_bomb=extra7)
        danger_base7 = self._danger_schedule(grid, bombs, players, horizon=horizon7)
        blast = self._blast_tiles(grid, bomb_pos[0], bomb_pos[1], radius)
        value = 0.0

        for _, epos, eradius, ebleft in enemies:
            dist = abs(epos[0] - bomb_pos[0]) + abs(epos[1] - bomb_pos[1])
            if epos not in blast and dist > radius + 2:
                continue

            base_count = self._survival_state_count(grid, epos, bombs, danger_base7, horizon=horizon7, cap=24)
            extra_count = self._survival_state_count(grid, epos, bombs, danger_extra7, horizon=horizon7, extra_bomb=extra7, cap=24)
            reduction = max(0, base_count - extra_count)

            local = 0.0
            if epos in blast and base_count > 0:
                if extra_count == 0:
                    local += 110
                elif extra_count <= 2:
                    local += 66 - 13 * extra_count
                elif extra_count <= 5:
                    local += 32 - 4 * extra_count
                local += min(24, 2.5 * reduction)
                # Faster chain explosions leave less time to dodge.
                if t7 < 7:
                    local += 10 * (7 - t7)
                # Low-mobility enemies are easier to convert into confirmed kills.
                local += max(0, 4 - self._mobility(grid, epos, bombs)) * 5
            elif reduction >= 4:
                # Not a direct hit, but the bomb meaningfully cuts reachable space.
                local += min(22, reduction * 2.5)

            if strict and local > 0:
                # Cautious one-step-evasion model. The enemy chooses a legal move in the
                # same tick before our bomb appears; high score only if most starts are still bad.
                starts = self._enemy_next_positions(grid, epos, bombs, danger_base7)
                extra6 = (bomb_pos, radius, self.agent_id, 6)
                h6 = 8
                danger_extra6 = self._danger_schedule(grid, bombs, players, horizon=h6, extra_bomb=extra6)
                danger_base6 = self._danger_schedule(grid, bombs, players, horizon=h6)
                doomed = 0
                small = 0
                for sp in starts:
                    bc = self._survival_state_count(grid, sp, bombs, danger_base6, horizon=h6, cap=16)
                    ec = self._survival_state_count(grid, sp, bombs, danger_extra6, horizon=h6, extra_bomb=extra6, cap=16)
                    if bc > 0 and ec == 0:
                        doomed += 1
                    if ec <= 3:
                        small += 1
                if starts:
                    frac_doom = doomed / len(starts)
                    frac_small = small / len(starts)
                    local *= (0.45 + 0.75 * frac_small)
                    if frac_doom >= 0.999:
                        local += 55
                    elif frac_doom >= 0.5:
                        local += 24 * frac_doom
                    else:
                        local -= 12

            value += local
        return value

    def _assassin_bomb_targets(self, grid, my_pos, radius, enemies, bombs, players, danger):
        if not enemies:
            return []
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}

        # Cheap candidate generation: only cells that could blast an observed or
        # one-step-predicted enemy position. Rank cheaply before any BFS-heavy kill eval.
        cheap = {}
        for _, epos, _, _ in enemies:
            enemy_mob = self._mobility(grid, epos, bombs)
            starts = [epos] + self._enemy_next_positions(grid, epos, bombs, danger)
            for sp in starts:
                for c in self._line_attack_cells(grid, sp, radius):
                    if c in bomb_positions:
                        continue
                    d_self = abs(c[0] - my_pos[0]) + abs(c[1] - my_pos[1])
                    if d_self > 9:
                        continue
                    d_enemy = abs(c[0] - epos[0]) + abs(c[1] - epos[1])
                    v = 42 - 2.0 * d_self - 1.0 * d_enemy + max(0, 4 - enemy_mob) * 7
                    if c == my_pos:
                        v += 10
                    if c not in cheap or v > cheap[c]:
                        cheap[c] = v

        prelim = sorted(cheap.items(), key=lambda x: x[1], reverse=True)[:5]
        scored = []
        for c, cheap_v in prelim:
            kv = self._candidate_kill_value(grid, c, radius, enemies, bombs, players, strict=False)
            if kv < 64:
                continue
            # Trust only cells from which a post-bomb escape exists. This is still
            # safe to compute because we kept the candidate list tiny.
            if not self._can_escape_after_bomb(grid, c, bombs, players, radius):
                continue
            value = min(90, kv) + 0.25 * cheap_v
            scored.append((c, value))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:3]

    def _enemy_future_bomb_risk(self, grid, pos, enemies, bombs):
        # Risk if an enemy can place a bomb on its current tile. This is not
        # immediate danger, but it is how strong agents trap predictable farmers.
        risk = 0
        mob = self._mobility(grid, pos, bombs)
        for _, epos, radius, bombs_left in enemies:
            if bombs_left <= 0:
                continue
            if pos in self._blast_tiles(grid, epos[0], epos[1], radius):
                d = abs(pos[0] - epos[0]) + abs(pos[1] - epos[1])
                risk += max(0, 9 - d) + max(0, 4 - mob) * 2
            elif abs(pos[0] - epos[0]) + abs(pos[1] - epos[1]) <= 1:
                risk += 4
        return risk

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
        score -= 0.35 * self._enemy_future_bomb_risk(grid, pos, enemies, bombs)
        if self.turn > 180:
            score -= 0.15 * (abs(pos[0] - 6) + abs(pos[1] - 6))
        if any(pos in danger.get(t, set()) for t in range(1, 5)):
            score -= 20
        return score
