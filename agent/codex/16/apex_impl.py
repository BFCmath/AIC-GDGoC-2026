"""
ApexHybridV2 — Definitive Bomberland agent for GDGoC 2026.
"""
from collections import deque

class Agent:
    team_id = "MetaApexClone2"
    MOVES = {0:(0,0), 1:(-1,0), 2:(1,0), 3:(0,-1), 4:(0,1)}

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.bomb_radius_mem = {}
        self.last_pos = None
        self.stuck = 0

    def act(self, obs: dict) -> int:
        try:
            self.turn += 1
            grid    = obs["map"]
            players = obs["players"]
            bombs   = obs["bombs"]

            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                return 0

            self._update_bomb_memory(bombs, players)

            me         = players[self.agent_id]
            my_pos     = (int(me[0]), int(me[1]))
            bombs_left = int(me[3])
            my_radius  = 1 + int(me[4])

            if self.last_pos == my_pos: self.stuck += 1
            else: self.stuck = 0
            self.last_pos = my_pos

            enemies = [
                (i, (int(p[0]), int(p[1])), 1 + int(p[4]), int(p[3]))
                for i, p in enumerate(players)
                if i != self.agent_id and int(p[2]) == 1
            ]

            danger = self._danger_schedule(grid, bombs, players, horizon=9)
            legal  = self._legal_actions(grid, my_pos, bombs, bombs_left)

            # ── 1. SURVIVAL (time-expanded) ─────────────────────────────────────
            threatened = any(my_pos in danger.get(t, set()) for t in range(1, 8))
            if threatened:
                escape = self._safe_bfs_action(grid, my_pos, bombs, players, danger, horizon=9)
                if escape is not None:
                    return escape
                safe_now = [a for a in legal if a != 5 and
                            self._next(my_pos, a) not in danger.get(1, set())]
                if safe_now:
                    return max(safe_now, key=lambda a:
                        self._mobility(grid, self._next(my_pos, a), bombs))
                return 0

            # ── 2. GRAB ITEM RIGHT NOW (cap > radius) ──────────────────────────
            immediate = []
            for a in legal:
                if a == 5: continue
                p = self._next(my_pos, a)
                cell = int(grid[p[0], p[1]])
                if cell in (3, 4) and p not in danger.get(1, set()):
                    score = (2 if cell == 4 else 1) + self._mobility(grid, p, bombs)
                    immediate.append((score, a))
            if immediate:
                return max(immediate)[1]

            # ── 3. BOMB PLACEMENT ───────────────────────────────────────────────
            if 5 in legal:
                boxes       = self._count_boxes_in_blast(grid, my_pos, my_radius)
                hit_enemies = self._enemies_in_blast(grid, my_pos, my_radius, enemies)
                trap        = self._trap_score(grid, my_pos, my_radius, enemies, bombs, players)

                value = boxes * 12 + len(hit_enemies) * 24 + trap

                if self.turn > 380 and enemies:
                    value += 10

                # Never place zero-box bombs unless kill/trap situation
                if boxes == 0 and not hit_enemies and trap < 8 and self.turn <= 380:
                    pass  # skip bomb
                elif value >= 10 and self._can_escape_after_bomb(grid, my_pos, bombs, players, my_radius):
                    escape_space = self._escape_space_after_bomb(grid, my_pos, bombs, players, my_radius)
                    if boxes >= 1 or hit_enemies or trap >= 4 or escape_space >= 1:
                        return 5

            # ── 4. BUILD TARGET LIST ────────────────────────────────────────────
            targets = []

            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    cell = int(grid[x, y])
                    if cell == 4:
                        v = 58 if bombs_left <= 1 else 38
                        targets.append(((x, y), v))
                    elif cell == 3:
                        v = 50 if my_radius <= 2 else 28
                        targets.append(((x, y), v))

            for pos, boxes_at in self._box_spots_detailed(grid, bombs, my_radius).items():
                mob = self._mobility(grid, pos, bombs)
                v = 16 + boxes_at * 15 + mob * 1.5
                targets.append((pos, v))

            for _, epos, eradius, ebleft in enemies:
                dist = abs(epos[0]-my_pos[0]) + abs(epos[1]-my_pos[1])
                if my_radius >= 2 or self.turn > 220:
                    v = max(8, 30 - 1.8*dist)
                    targets.append((epos, v))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    for k in range(1, min(my_radius+2, 5)):
                        tp = (epos[0]+dx*k, epos[1]+dy*k)
                        if self._passable(grid, tp[0], tp[1]):
                            kill_v = 35 if k <= my_radius else 22
                            targets.append((tp, kill_v))
                            break

            move = self._best_move_to_targets(grid, my_pos, bombs, players, danger, targets, enemies)
            if move is not None:
                return move

            # ── 5. MOBILE SAFE FALLBACK ─────────────────────────────────────────
            candidates = [a for a in legal if a != 5 and
                          self._next(my_pos, a) not in danger.get(1, set())]
            if not candidates:
                return 0
            if self.stuck >= 3:
                movers = [a for a in candidates if a != 0]
                if movers: candidates = movers
            return max(candidates, key=lambda a:
                self._fallback_score(grid, self._next(my_pos, a), bombs, enemies, danger))

        except Exception:
            return 0

    # ── BOMB MEMORY ──────────────────────────────────────────────────────────────
    def _update_bomb_memory(self, bombs, players):
        seen = set()
        for b in bombs:
            x, y, owner = int(b[0]), int(b[1]), int(b[3])
            key = (x, y, owner)
            seen.add(key)
            if key not in self.bomb_radius_mem:
                r = 1 + int(players[owner][4]) if 0 <= owner < len(players) else 2
                self.bomb_radius_mem[key] = max(1, min(5, r))
        for k in list(self.bomb_radius_mem):
            if k not in seen: del self.bomb_radius_mem[k]

    def _radius_for_bomb(self, b, players):
        key = (int(b[0]), int(b[1]), int(b[3]))
        if key in self.bomb_radius_mem: return self.bomb_radius_mem[key]
        owner = int(b[3])
        return 1 + int(players[owner][4]) if 0 <= owner < len(players) else 2

    # ── GEOMETRY ─────────────────────────────────────────────────────────────────
    def _next(self, pos, action):
        dx, dy = self.MOVES.get(action, (0,0))
        return (pos[0]+dx, pos[1]+dy)

    def _in_bounds(self, g, x, y):
        return 0 <= x < g.shape[0] and 0 <= y < g.shape[1]

    def _passable(self, g, x, y):
        return self._in_bounds(g, x, y) and int(g[x, y]) in (0, 3, 4)

    def _blast_tiles(self, g, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            for r in range(1, radius+1):
                x, y = bx+dx*r, by+dy*r
                if not self._in_bounds(g, x, y): break
                cell = int(g[x, y])
                if cell == 1: break
                tiles.add((x, y))
                if cell == 2: break
        return tiles

    def _legal_actions(self, g, pos, bombs, bombs_left):
        bp = {(int(b[0]), int(b[1])) for b in bombs}
        actions = [0]
        for a in [1,2,3,4]:
            nx, ny = self._next(pos, a)
            if self._passable(g, nx, ny) and (nx, ny) not in bp:
                actions.append(a)
        if bombs_left > 0 and pos not in bp:
            actions.append(5)
        return actions

    def _mobility(self, g, pos, bombs):
        bp = {(int(b[0]), int(b[1])) for b in bombs}
        return sum(1 for a in [1,2,3,4]
                   if self._passable(g, *self._next(pos, a))
                   and self._next(pos, a) not in bp)

    # ── TIME-EXPANDED DANGER ─────────────────────────────────────────────────────
    def _bomb_explosion_times(self, g, bombs, players, extra_bomb=None):
        bl = []
        for b in bombs:
            bl.append({"pos":(int(b[0]),int(b[1])), "timer":max(1,int(b[2])),
                       "radius":self._radius_for_bomb(b, players)})
        if extra_bomb:
            pos, radius, owner, timer = extra_bomb
            bl.append({"pos":pos, "timer":max(1,int(timer)), "radius":radius})
        times = [b["timer"] for b in bl]
        changed = True
        while changed:
            changed = False
            for i, b in enumerate(bl):
                blast = self._blast_tiles(g, b["pos"][0], b["pos"][1], b["radius"])
                for j, other in enumerate(bl):
                    if i != j and other["pos"] in blast and times[j] > times[i]:
                        times[j] = times[i]; changed = True
        return bl, times

    def _danger_schedule(self, g, bombs, players, horizon=9, extra_bomb=None):
        sched = {t: set() for t in range(1, horizon+1)}
        bl, times = self._bomb_explosion_times(g, bombs, players, extra_bomb)
        for b, t in zip(bl, times):
            if 1 <= t <= horizon:
                sched[t].update(self._blast_tiles(g, b["pos"][0], b["pos"][1], b["radius"]))
        return sched

    def _bomb_pos_alive_at(self, bombs, t, extra=None):
        out = {(int(b[0]), int(b[1])) for b in bombs if int(b[2]) > t}
        if extra:
            pos, _, _, timer = extra
            if int(timer) > t: out.add(pos)
        return out

    # ── TIME-EXPANDED BFS ─────────────────────────────────────────────────────────
    def _safe_bfs_action(self, g, start, bombs, players, danger, horizon=9,
                          extra_bomb=None, require_leave_blast=None):
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None; best_score = -10**9
        while q:
            pos, t, first = q.popleft()
            future_bad = any(pos in danger.get(tt, set()) for tt in range(t+1, horizon+1))
            if t > 0 and not future_bad:
                score = -t + 2*self._mobility(g, pos, bombs)
                if require_leave_blast and pos not in require_leave_blast:
                    score += 10
                if score > best_score:
                    best_score = score; best = first
                    if t <= 3 and (not require_leave_blast or pos not in require_leave_blast):
                        return first
            if t >= horizon: continue
            bn = self._bomb_pos_alive_at(bombs, t+1, extra_bomb)
            for a in [0,1,2,3,4]:
                npos = self._next(pos, a)
                if not self._passable(g, npos[0], npos[1]): continue
                if npos in bn and npos != start: continue
                if npos in danger.get(t+1, set()): continue
                state = (npos, t+1)
                if state in seen: continue
                seen.add(state)
                q.append((npos, t+1, a if first is None else first))
        return best

    def _best_move_to_targets(self, g, start, bombs, players, danger, targets, enemies):
        if not targets: return None
        tv = {}
        for pos, value in targets:
            if pos not in tv or value > tv[pos]: tv[pos] = value
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None; best_score = -10**9
        while q:
            pos, t, first = q.popleft()
            if t > 0 and pos in tv:
                score = (tv[pos] - 1.85*t
                         + 0.7*self._mobility(g, pos, bombs)
                         - 0.65*self._enemy_bomb_threat(g, pos, enemies))
                if score > best_score:
                    best_score = score; best = first
            if t >= 12: continue
            bn = self._bomb_pos_alive_at(bombs, t+1)
            for a in [0,1,2,3,4]:
                npos = self._next(pos, a)
                if not self._passable(g, npos[0], npos[1]): continue
                if npos in bn: continue
                if npos in danger.get(t+1, set()): continue
                state = (npos, t+1)
                if state in seen: continue
                seen.add(state)
                q.append((npos, t+1, a if first is None else first))
        return best if best is not None and best_score > -8 else None

    # ── BOMB HELPERS ─────────────────────────────────────────────────────────────
    def _can_escape_after_bomb(self, g, pos, bombs, players, radius):
        extra  = (pos, radius, self.agent_id, 7)
        danger = self._danger_schedule(g, bombs, players, horizon=9, extra_bomb=extra)
        blast  = self._blast_tiles(g, pos[0], pos[1], radius)
        return self._safe_bfs_action(g, pos, bombs, players, danger, horizon=9,
                                     extra_bomb=extra, require_leave_blast=blast) is not None

    def _escape_space_after_bomb(self, g, pos, bombs, players, radius):
        extra   = (pos, radius, self.agent_id, 7)
        danger  = self._danger_schedule(g, bombs, players, horizon=9, extra_bomb=extra)
        blocked = self._bomb_pos_alive_at(bombs, 1, extra)
        return sum(1 for a in [1,2,3,4]
                   if self._passable(g, *self._next(pos, a))
                   and self._next(pos, a) not in blocked
                   and self._next(pos, a) not in danger.get(1, set()))

    def _count_boxes_in_blast(self, g, pos, radius):
        return sum(1 for x,y in self._blast_tiles(g, pos[0], pos[1], radius)
                   if int(g[x,y]) == 2)

    def _enemies_in_blast(self, g, pos, radius, enemies):
        blast = self._blast_tiles(g, pos[0], pos[1], radius)
        return [eid for eid, epos, _, _ in enemies if epos in blast]

    def _trap_score(self, g, pos, radius, enemies, bombs, players):
        if not enemies: return 0
        blast = self._blast_tiles(g, pos[0], pos[1], radius)
        score = 0
        bp = {(int(b[0]), int(b[1])) for b in bombs}
        for _, epos, _, _ in enemies:
            if epos in blast:
                exits = sum(1 for a in [1,2,3,4]
                            if self._passable(g, *self._next(epos, a))
                            and self._next(epos, a) not in blast
                            and self._next(epos, a) not in bp)
                score += max(0, 22 - 6*exits)
            dist = abs(epos[0]-pos[0]) + abs(epos[1]-pos[1])
            if dist <= radius+1:
                score += max(0, 8-dist)
        return score

    def _box_spots_detailed(self, g, bombs, my_radius):
        bp = {(int(b[0]), int(b[1])) for b in bombs}
        spots = {}
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                if int(g[x, y]) != 2: continue
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if not self._passable(g, nx, ny): continue
                    if (nx, ny) in bp: continue
                    p = (nx, ny)
                    boxes_at = self._count_boxes_in_blast(g, p, my_radius)
                    if p not in spots or spots[p] < boxes_at:
                        spots[p] = boxes_at
        return spots

    def _enemy_bomb_threat(self, g, pos, enemies):
        penalty = 0
        for _, epos, radius, bleft in enemies:
            if bleft <= 0: continue
            if pos in self._blast_tiles(g, epos[0], epos[1], radius):
                d = abs(pos[0]-epos[0]) + abs(pos[1]-epos[1])
                penalty += max(0, 6-d)
        return penalty

    def _fallback_score(self, g, pos, bombs, enemies, danger):
        score  = 2 * self._mobility(g, pos, bombs)
        score -= self._enemy_bomb_threat(g, pos, enemies)
        if self.turn > 180:
            score -= 0.12 * (abs(pos[0]-6) + abs(pos[1]-6))
        if any(pos in danger.get(t, set()) for t in range(1, 5)):
            score -= 30
        return score