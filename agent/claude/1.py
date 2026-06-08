"""
ChampionAgent v3 — Elite Bomberland AI
========================================

Improvements over v2:
1. TIME-GATED APPROACH BFS: hunt enemies even when path goes near danger,
   as long as we arrive before bombs explode (danger_timer[cell] > depth)
2. Lowered bomb thresholds: bombs placed more aggressively in mid/late phase
3. Dual-mode bomb check: box-value for early, enemy-value for mid/late
4. Correct escape BFS: safe = cell NOT in any blast zone, reached before explosion
5. Chain reaction model: predict cascading bomb explosions
6. Anti-loop with directional bias away from recent positions
"""

from collections import deque


class ChampionAgent:
    MOVES = {0:(0,0), 1:(-1,0), 2:(1,0), 3:(0,-1), 4:(0,1)}
    team_id = "ChampionAgent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.history   = deque(maxlen=20)
        self.step      = 0

    # ── MAIN ─────────────────────────────────────────────────────────

    def act(self, obs):
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        self.step += 1

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos     = (int(my_x), int(my_y))
        my_radius  = max(1, int(bomb_bonus) + 1)
        bombs_left = int(bombs_left)
        n_alive    = sum(1 for p in players if p[2] == 1)
        phase      = self._phase(n_alive)

        self.history.append(my_pos)

        bomb_pos_set = {(int(b[0]), int(b[1])) for b in bombs}
        blocked      = bomb_pos_set - {my_pos}

        alive_enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]

        # ── Danger model ──────────────────────────────────────────────
        danger_timer = self._build_danger_timer(grid, bombs, players)
        in_danger    = my_pos in danger_timer

        # ── ESCAPE ───────────────────────────────────────────────────
        if in_danger:
            esc = self._escape_bfs(grid, my_pos, blocked, danger_timer)
            if esc is not None:
                return esc
            return self._panic_move(grid, my_pos, blocked, danger_timer)

        # ── BOMB PLACEMENT ───────────────────────────────────────────
        if bombs_left > 0 and my_pos not in bomb_pos_set:
            should_bomb, reason = self._should_bomb(
                grid, my_pos, my_radius, alive_enemies,
                danger_timer, phase, bombs, players
            )
            if should_bomb:
                return 5

        # ── ITEM COLLECTION ──────────────────────────────────────────
        if phase in ('early', 'mid'):
            item_tiles = {
                (x, y)
                for x in range(grid.shape[0])
                for y in range(grid.shape[1])
                if grid[x, y] in [3, 4]
            }
            if item_tiles:
                mv = self._bfs_time_gated(
                    grid, my_pos, item_tiles, blocked, danger_timer, max_dist=8
                )
                if mv is not None:
                    return mv

        # ── PHASE OBJECTIVES ─────────────────────────────────────────
        if phase == 'early':
            spots = self._best_box_spots(grid, blocked)
            if spots:
                mv = self._bfs_time_gated(
                    grid, my_pos, spots, blocked, danger_timer
                )
                if mv is not None:
                    return mv

        elif phase in ('mid', 'late'):
            if alive_enemies:
                targets = sorted(
                    alive_enemies,
                    key=lambda e: abs(e[0]-my_pos[0]) + abs(e[1]-my_pos[1])
                )
                mv = self._bfs_time_gated(
                    grid, my_pos, set(targets[:2]), blocked, danger_timer
                )
                if mv is not None:
                    return mv

        # ── BOX FARMING FALLBACK ─────────────────────────────────────
        spots = self._box_adjacent_spots(grid, blocked)
        if spots:
            mv = self._bfs_time_gated(
                grid, my_pos, spots, blocked, danger_timer
            )
            if mv is not None:
                return mv

        # ── ANTI-LOOP MOVE ───────────────────────────────────────────
        return self._explore(grid, my_pos, blocked, danger_timer)

    # ── DANGER MODEL ─────────────────────────────────────────────────

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            for r in range(1, radius+1):
                x, y = bx+dx*r, by+dy*r
                if not self._inb(grid, x, y): break
                c = grid[x, y]
                if c == 1: break
                tiles.add((x, y))
                if c == 2: break
        return tiles

    def _build_danger_timer(self, grid, bombs, players):
        """danger_timer[pos] = min steps until a bomb covering pos explodes."""
        dt = {}
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner = int(b[3]) if len(b) > 3 else 0
            r = max(1, int(players[owner][4])+1) if owner < len(players) else 1
            for pos in self._blast_tiles(grid, bx, by, r):
                if pos not in dt or timer < dt[pos]:
                    dt[pos] = timer
        return dt

    def _build_danger_timer_with_bomb(self, grid, danger_timer, bx, by,
                                       radius, bombs, players):
        """Add a new bomb at (bx,by) with timer=7, propagate chain reactions."""
        dt = dict(danger_timer)
        timer_new = 7
        for pos in self._blast_tiles(grid, bx, by, radius):
            if pos not in dt or timer_new < dt[pos]:
                dt[pos] = timer_new

        # Chain reaction: if new blast hits existing bomb, that bomb explodes sooner
        changed = True
        while changed:
            changed = False
            for b in bombs:
                bbx, bby, btimer = int(b[0]), int(b[1]), int(b[2])
                bowner = int(b[3]) if len(b) > 3 else 0
                br = max(1, int(players[bowner][4])+1) if bowner < len(players) else 1
                bpos = (bbx, bby)
                if bpos in dt:
                    chain_t = dt[bpos]  # this bomb detonates at chain_t
                    for pos in self._blast_tiles(grid, bbx, bby, br):
                        if pos not in dt or chain_t < dt[pos]:
                            dt[pos] = chain_t
                            changed = True
        return dt

    # ── ESCAPE BFS ───────────────────────────────────────────────────

    def _escape_bfs(self, grid, start, blocked, danger_timer):
        """
        BFS with depth tracking. We start at depth=0 in danger.
        A destination cell is safe if danger_timer[cell] > depth_to_reach
        (bomb hasn't exploded yet) OR cell is not in danger_timer at all.
        Key invariant: a cell is TRULY safe only if NOT in danger_timer.
        We route through dangerous cells only if bomb_timer > arrival_depth.
        """
        q = deque([(start, 0, None)])
        seen = {start}
        while q:
            pos, depth, first_action = q.popleft()
            # Arrived at this cell — is it permanently safe?
            if depth > 0 and pos not in danger_timer:
                return first_action
            if depth >= 14:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if not self._passable(grid, nx, ny): continue
                if npos in blocked: continue
                if npos in seen: continue
                nd = depth + 1
                # Can we safely pass through npos?
                nt = danger_timer.get(npos, 999)
                if nt <= nd: continue  # would be hit when we arrive
                seen.add(npos)
                q.append((npos, nd, a if first_action is None else first_action))
        return None

    def _panic_move(self, grid, pos, blocked, danger_timer):
        """Move to neighbor with maximum time-to-explosion."""
        best, best_t = None, -1
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(pos, a)
            if not self._passable(grid, nx, ny) or (nx,ny) in blocked: continue
            t = danger_timer.get((nx,ny), 999)
            if t > best_t:
                best_t, best = t, a
        return best if best is not None else 0

    # ── BOMB DECISION ────────────────────────────────────────────────

    def _should_bomb(self, grid, my_pos, my_radius, alive_enemies,
                      danger_timer, phase, bombs, players):
        blast = self._blast_tiles(grid, my_pos[0], my_pos[1], my_radius)
        enemies_hit = sum(1 for e in alive_enemies if e in blast)
        boxes_hit   = sum(1 for (x,y) in blast if grid[x,y] == 2)

        # Score thresholds by phase
        if phase == 'early':
            worth_it = boxes_hit >= 1 or enemies_hit >= 1
        elif phase == 'mid':
            worth_it = enemies_hit >= 1 or boxes_hit >= 2
        else:  # late
            worth_it = enemies_hit >= 1

        if not worth_it:
            return False, 'no_value'

        # Escape check after bomb
        dt_new = self._build_danger_timer_with_bomb(
            grid, danger_timer, my_pos[0], my_pos[1], my_radius, bombs, players
        )
        blocked_for_escape = {(int(b[0]), int(b[1])) for b in bombs} - {my_pos}
        # Can we escape from my_pos given the new danger?
        esc = self._escape_bfs(grid, my_pos, blocked_for_escape, dt_new)
        if esc is None:
            return False, 'no_escape'
        return True, 'bomb!'

    # ── TIME-GATED APPROACH BFS ──────────────────────────────────────

    def _bfs_time_gated(self, grid, start, targets, blocked, danger_timer,
                         max_dist=40):
        """
        BFS toward targets. Allows passing through dangerous cells ONLY if
        danger_timer[cell] > depth_to_reach (we arrive before bomb explodes).
        This enables hunting enemies even when bombs are between us.
        """
        q = deque([(start, None, 0)])
        seen = {start}
        while q:
            pos, fa, dist = q.popleft()
            if pos in targets and fa is not None:
                return fa
            if dist >= max_dist:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen: continue
                if not self._passable(grid, nx, ny): continue
                if npos in blocked: continue
                nd = dist + 1
                nt = danger_timer.get(npos, 999)
                if nt <= nd: continue  # would be in explosion when we arrive
                seen.add(npos)
                q.append((npos, a if fa is None else fa, nd))
        return None

    # ── STRATEGY HELPERS ────────────────────────────────────────────

    def _phase(self, n_alive):
        if self.step < 80 or n_alive == 4:
            return 'early'
        elif self.step < 280 or n_alive >= 3:
            return 'mid'
        else:
            return 'late'

    def _best_box_spots(self, grid, blocked):
        value = {}
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 2: continue
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if self._passable(grid,nx,ny) and (nx,ny) not in blocked:
                        value[(nx,ny)] = value.get((nx,ny),0)+1
        if not value: return set()
        best_v = max(value.values())
        return {p for p,v in value.items() if v >= max(best_v-1, 1)}

    def _box_adjacent_spots(self, grid, blocked):
        spots = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x,y] != 2: continue
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if self._passable(grid,nx,ny) and (nx,ny) not in blocked:
                        spots.add((nx,ny))
        return spots

    def _explore(self, grid, my_pos, blocked, danger_timer):
        recent = list(self.history)
        candidates = []
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(my_pos, a)
            npos = (nx, ny)
            if not self._passable(grid, nx, ny): continue
            if npos in blocked: continue
            nt = danger_timer.get(npos, 999)
            if nt <= 1: continue  # imminent
            recency = recent.count(npos)
            # Also prefer cells farther from recent danger
            danger_factor = 0 if npos not in danger_timer else (7 - danger_timer[npos])
            candidates.append((recency + danger_factor, a))
        if candidates:
            candidates.sort()
            return candidates[0][1]
        # Last resort: any valid move
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(my_pos, a)
            if self._passable(grid, nx, ny) and (nx,ny) not in blocked:
                return a
        return 0

    # ── UTILS ───────────────────────────────────────────────────────

    def _next_pos(self, pos, a):
        dx, dy = self.MOVES[a]
        return pos[0]+dx, pos[1]+dy

    def _inb(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._inb(grid, x, y) and grid[x, y] in [0, 3, 4]