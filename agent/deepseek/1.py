import numpy as np
from collections import deque

class Agent:
    team_id = "NemesisTacticianV1"

    MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.last_pos = None
        self.stuck = 0
        self.bomb_radius_mem = {}          # (x, y, owner) -> radius at first sight
        self.visited_cells = set()         # for patrol diversity

    # ---------------------------------------------------------------
    #  Main entry point
    # ---------------------------------------------------------------
    def act(self, obs: dict) -> int:
        try:
            self.turn += 1
            grid = obs["map"]
            players = obs["players"]
            bombs = obs["bombs"]

            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                return 0

            me = players[self.agent_id]
            my_pos = (int(me[0]), int(me[1]))
            bombs_left = int(me[3])
            radius = 1 + int(me[4])

            # track stuckness
            if self.last_pos == my_pos:
                self.stuck += 1
            else:
                self.stuck = 0
            self.last_pos = my_pos

            # update bomb radius memory
            self._update_bomb_memory(bombs, players)

            # danger schedule for existing bombs
            danger = self._danger_schedule(grid, bombs, players, horizon=10)
            blast_now = danger.get(1, set())   # explosions that happen after this step

            # legal actions
            legal = self._legal_actions(grid, my_pos, bombs, bombs_left)

            # --- Immediate survival ---
            if my_pos in blast_now:
                escape = self._escape_now(grid, my_pos, bombs, players, blast_now, danger)
                if escape is not None:
                    return escape
                if 5 in legal:
                    return 5
                return 0

            # --- Safe item pickup ---
            item_move = self._grab_item_safe(grid, my_pos, legal, danger, bombs, players)
            if item_move is not None:
                return item_move

            # --- Bomb placement ---
            bomb_action = self._consider_bomb(grid, my_pos, players, bombs, bombs_left, radius, danger, legal)
            if bomb_action is not None:
                return bomb_action

            # --- Go to best farming/tactical target ---
            target_action = self._go_to_targets(grid, my_pos, players, bombs, radius, danger, bombs_left)
            if target_action is not None:
                return target_action

            # --- Safe fallback with mobility ---
            return self._safe_fallback(grid, my_pos, legal, danger, players, bombs)

        except Exception:
            return 0

    # ---------------------------------------------------------------
    #  State models (danger, bomb memory)
    # ---------------------------------------------------------------
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
        # remove entries for bombs no longer on board
        for key in list(self.bomb_radius_mem.keys()):
            if key not in seen:
                del self.bomb_radius_mem[key]

    def _radius_for_bomb(self, b, players):
        key = (int(b[0]), int(b[1]), int(b[3]))
        if key in self.bomb_radius_mem:
            return self.bomb_radius_mem[key]
        owner = int(b[3])
        return 1 + int(players[owner][4]) if 0 <= owner < len(players) else 2

    def _bomb_explosion_times(self, grid, bombs, players, extra_bomb=None):
        """
        Returns (bomb_list, times) where times are the final explosion step index
        (1 = explodes in the upcoming engine step). Chain reactions are propagated.
        """
        bomb_list = []
        for b in bombs:
            bomb_list.append({
                "pos": (int(b[0]), int(b[1])),
                "timer": max(1, int(b[2])),   # timer 0 becomes 1 for schedule
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

    def _danger_schedule(self, grid, bombs, players, horizon=10, extra_bomb=None):
        """Returns dict {t: set of cells} for t=1..horizon where explosion occurs."""
        schedule = {t: set() for t in range(1, horizon + 1)}
        bomb_list, times = self._bomb_explosion_times(grid, bombs, players, extra_bomb)
        for b, t in zip(bomb_list, times):
            if 1 <= t <= horizon:
                schedule[t].update(self._blast_tiles(grid, b["pos"][0], b["pos"][1], b["radius"]))
        return schedule

    # ---------------------------------------------------------------
    #  Geometry helpers
    # ---------------------------------------------------------------
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

    def _bomb_positions_alive_at(self, bombs, t, extra_bomb=None):
        """Bombs that still exist after t engine steps (i.e. original timer > t)."""
        out = {(int(b[0]), int(b[1])) for b in bombs if int(b[2]) > t}
        if extra_bomb is not None:
            pos, _, _, timer = extra_bomb
            if int(timer) > t:
                out.add(pos)
        return out

    # ---------------------------------------------------------------
    #  Action legality
    # ---------------------------------------------------------------
    def _legal_actions(self, grid, my_pos, bombs, bombs_left):
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        actions = [0]
        for a in [1, 2, 3, 4]:
            np = self._next(my_pos, a)
            if self._passable(grid, np[0], np[1]) and np not in bomb_positions:
                actions.append(a)
        if bombs_left > 0 and my_pos not in bomb_positions:
            actions.append(5)
        return actions

    # ---------------------------------------------------------------
    #  Immediate escape from blast_now
    # ---------------------------------------------------------------
    def _escape_now(self, grid, start, bombs, players, blast_now, danger):
        # only consider moves that leave blast_now
        legal = self._legal_actions(grid, start, bombs, 999)  # we only need movement
        safe_moves = [a for a in legal if a != 5 and self._next(start, a) not in blast_now]
        if not safe_moves:
            return None
        # choose the one that gives best future safety (time‑expanded)
        best, best_score = None, -1e9
        for a in safe_moves:
            np = self._next(start, a)
            future_safe = self._safe_bfs_from(grid, np, bombs, players, danger, depth=8)
            if future_safe is not None:
                score = future_safe  # number of steps safe (higher = better)
                if score > best_score:
                    best_score = score
                    best = a
        return best if best is not None else safe_moves[0]

    def _safe_bfs_from(self, grid, start, bombs, players, danger, depth=8):
        """Returns the number of consecutive safe steps (or -1 if no danger at all)."""
        # BFS that finds the longest horizon without future danger
        q = deque([(start, 0)])
        seen = {start}
        max_safe = 0
        while q:
            pos, t = q.popleft()
            future_bad = any(pos in danger.get(tt, set()) for tt in range(t+1, depth+1))
            if not future_bad:
                max_safe = max(max_safe, t)
                continue
            if t >= depth:
                continue
            for a in [1,2,3,4]:
                np = self._next(pos, a)
                if not self._passable(grid, np[0], np[1]):
                    continue
                if np in seen:
                    continue
                if np in self._bomb_positions_alive_at(bombs, t+1):
                    continue
                if np in danger.get(t+1, set()):
                    continue
                seen.add(np)
                q.append((np, t+1))
        return max_safe

    # ---------------------------------------------------------------
    #  Safe item grab
    # ---------------------------------------------------------------
    def _grab_item_safe(self, grid, my_pos, legal, danger, bombs, players):
        immediate = []
        for a in legal:
            if a == 5:
                continue
            np = self._next(my_pos, a)
            if int(grid[np[0], np[1]]) in (3, 4) and np not in danger.get(1, set()):
                immediate.append((a, np))
        if not immediate:
            return None
        # prefer capacity if bombs_left <= 1, then radius if radius <= 2
        best = None
        best_score = -1
        for a, np in immediate:
            cell = int(grid[np[0], np[1]])
            score = 0
            if cell == 4 and bombs_left <= 1:
                score = 100
            elif cell == 3 and radius <= 2:
                score = 80
            else:
                score = 50
            # quick check: can I escape after grabbing?
            if not self._safe_bfs_from(grid, np, bombs, players, danger, depth=4):
                score -= 30
            if score > best_score:
                best_score = score
                best = a
        return best

    # ---------------------------------------------------------------
    #  Bomb consideration
    # ---------------------------------------------------------------
    def _consider_bomb(self, grid, my_pos, players, bombs, bombs_left, radius, danger, legal):
        if 5 not in legal:
            return None
        # compute value
        boxes = sum(1 for x,y in self._blast_tiles(grid, my_pos[0], my_pos[1], radius) if int(grid[x,y])==2)
        enemies = self._enemy_state(players)
        hit = self._enemies_hit(grid, my_pos, radius, enemies)
        trap = self._trap_score(grid, my_pos, radius, enemies, bombs, players)
        expected_items = boxes * 0.6
        value = boxes * 12 + len(hit) * 24 + trap + expected_items * 8

        # late game tie‑breaker: bomb even if no box hit, as long as safe
        if self.turn > 350 and boxes == 0 and not hit and trap == 0:
            value = 4   # small encouragement for bombs_placed

        threshold = 10 if self.turn <= 350 else 5
        if value < threshold:
            return None

        # escape check with time‑expanded danger
        if not self._can_escape_after_bomb(grid, my_pos, bombs, players, radius):
            return None

        # extra: don't bomb if a very valuable item is right next to us (we already handled pickup)
        return 5

    def _can_escape_after_bomb(self, grid, my_pos, bombs, players, radius):
        extra = (my_pos, radius, self.agent_id, 7)
        danger_with = self._danger_schedule(grid, bombs, players, horizon=10, extra_bomb=extra)
        blast = self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
        return self._safe_bfs_leave_blast(grid, my_pos, bombs, players, danger_with, blast, horizon=10) is not None

    def _safe_bfs_leave_blast(self, grid, start, bombs, players, danger, blast, horizon=10):
        """Return first action that leads to a cell outside the blast and safe for entire horizon."""
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        blocked = self._bomb_positions_alive_at(bombs, 1, extra_bomb=(start, 999, 0, 7))
        # start cell after bomb will be blocked? No, moving off it is allowed; but we must not step onto another bomb.
        for a in [1,2,3,4]:
            np = self._next(start, a)
            if not self._passable(grid, np[0], np[1]):
                continue
            if np in blocked:
                continue
            if np in danger.get(1, set()):
                continue
            state = (np, 1)
            if state in seen:
                continue
            seen.add(state)
            q.append((np, 1, a))
        while q:
            pos, t, first = q.popleft()
            # must be outside blast at t=1 (already checked when added) and safe for rest of horizon
            future_bad = any(pos in danger.get(tt, set()) for tt in range(t+1, horizon+1))
            if pos not in blast and not future_bad:
                return first
            if t >= horizon:
                continue
            blocked_next = self._bomb_positions_alive_at(bombs, t+1, extra_bomb=(start, 999, 0, 7))
            for a in [1,2,3,4]:
                np = self._next(pos, a)
                if not self._passable(grid, np[0], np[1]):
                    continue
                if np in blocked_next:
                    continue
                if np in danger.get(t+1, set()):
                    continue
                state = (np, t+1)
                if state in seen:
                    continue
                seen.add(state)
                q.append((np, t+1, first))
        return None

    # ---------------------------------------------------------------
    #  Target scoring and BFS to best target
    # ---------------------------------------------------------------
    def _go_to_targets(self, grid, my_pos, players, bombs, radius, danger, bombs_left):
        enemies = self._enemy_state(players)
        targets = []   # list of (pos, score)
        # items
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = int(grid[x, y])
                if cell == 4:    # capacity
                    s = 65 if bombs_left <= 1 else 40
                    targets.append(((x, y), s))
                elif cell == 3:  # radius
                    s = 55 if radius <= 2 else 30
                    targets.append(((x, y), s))

        # box spots
        for pos, box_count in self._box_spots(grid, bombs).items():
            boxes_destroyed = self._count_boxes_in_blast(grid, pos, radius)
            expected_items = boxes_destroyed * 0.6
            score = 15 + boxes_destroyed * 14 + box_count * 2 + expected_items * 8
            targets.append((pos, score))

        # enemy approach (if we have bombs and radius)
        if bombs_left > 0 and radius >= 2:
            for eid, epos, eradius, ebombs in enemies:
                # cells from which we can hit enemy if we bomb there
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    for r in range(1, radius+1):
                        p = (epos[0]+dx*r, epos[1]+dy*r)
                        if not self._in_bounds(grid, p[0], p[1]):
                            break
                        cell = int(grid[p[0], p[1]])
                        if cell == 1:
                            break
                        if cell == 2:
                            break
                        if self._passable(grid, p[0], p[1]) and p not in {(int(b[0]),int(b[1])) for b in bombs}:
                            dist = abs(p[0]-my_pos[0])+abs(p[1]-my_pos[1])
                            score = 22 - 1.5 * dist
                            targets.append((p, score))

        if not targets:
            return None

        # unique best scores
        target_score = {}
        for pos, s in targets:
            if pos not in target_score or s > target_score[pos]:
                target_score[pos] = s

        # time‑expanded BFS
        return self._best_move_to_targets(grid, my_pos, bombs, players, danger, target_score)

    def _best_move_to_targets(self, grid, start, bombs, players, danger, target_score):
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None
        best_score = -1e9
        while q:
            pos, t, first = q.popleft()
            if t > 0 and pos in target_score:
                s = target_score[pos] - 1.8 * t + 0.5 * self._mobility(grid, pos, bombs)
                s -= self._enemy_bomb_threat(grid, pos, self._enemy_state(players))
                if s > best_score:
                    best_score = s
                    best = first
            if t >= 12:
                continue
            blocked_next = self._bomb_positions_alive_at(bombs, t+1)
            for a in [1,2,3,4]:
                np = self._next(pos, a)
                if not self._passable(grid, np[0], np[1]):
                    continue
                if np in blocked_next:
                    continue
                if np in danger.get(t+1, set()):
                    continue
                state = (np, t+1)
                if state in seen:
                    continue
                seen.add(state)
                q.append((np, t+1, a if first is None else first))
        return best if best is not None and best_score > -8 else None

    # ---------------------------------------------------------------
    #  Fallback move
    # ---------------------------------------------------------------
    def _safe_fallback(self, grid, my_pos, legal, danger, players, bombs):
        candidates = [a for a in legal if a != 5 and self._next(my_pos, a) not in danger.get(1, set())]
        if not candidates:
            return 0
        if self.stuck >= 3:
            movers = [a for a in candidates if a != 0]
            if movers:
                candidates = movers
        # score each candidate cell
        enemies = self._enemy_state(players)
        def score_cell(a):
            pos = self._next(my_pos, a)
            s = 2 * self._mobility(grid, pos, bombs)
            s -= self._enemy_bomb_threat(grid, pos, enemies)
            if self.turn > 180:
                s -= 0.1 * (abs(pos[0]-6) + abs(pos[1]-6))
            # patrol: slight bonus for less recently visited grass cells
            if pos not in self.visited_cells and int(grid[pos[0], pos[1]]) == 0:
                s += 1
            # avoid future danger
            if any(pos in danger.get(t, set()) for t in range(2, 5)):
                s -= 20
            return s
        best = max(candidates, key=score_cell)
        # update visited cells (keep last 40)
        self.visited_cells.add(self._next(my_pos, best))
        if len(self.visited_cells) > 40:
            self.visited_cells.pop()
        return best

    # ---------------------------------------------------------------
    #  Various scoring helpers
    # ---------------------------------------------------------------
    def _enemy_state(self, players):
        return [(i, (int(p[0]), int(p[1])), 1+int(p[4]), int(p[3]))
                for i, p in enumerate(players) if i != self.agent_id and int(p[2]) == 1]

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for x,y in self._blast_tiles(grid, pos[0], pos[1], radius) if int(grid[x,y]) == 2)

    def _enemies_hit(self, grid, pos, radius, enemies):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        hit = []
        for eid, epos, _, _ in enemies:
            if epos in blast:
                hit.append(eid)
        return hit

    def _trap_score(self, grid, pos, radius, enemies, bombs, players):
        if not enemies:
            return 0
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        score = 0
        for _, epos, _, _ in enemies:
            if epos in blast:
                # penalty based on their mobility
                score += max(0, 16 - 2 * self._mobility(grid, epos, bombs))
            dist = abs(epos[0]-pos[0]) + abs(epos[1]-pos[1])
            if dist <= radius + 1:
                score += max(0, 6 - dist)
        return score

    def _box_spots(self, grid, bombs):
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        spots = {}
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if int(grid[x,y]) != 2:
                    continue
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in bomb_positions:
                        spots[(nx, ny)] = spots.get((nx, ny), 0) + 1
        return spots

    def _mobility(self, grid, pos, bombs):
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        cnt = 0
        for a in [1,2,3,4]:
            np = self._next(pos, a)
            if self._passable(grid, np[0], np[1]) and np not in bomb_positions:
                cnt += 1
        return cnt

    def _enemy_bomb_threat(self, grid, pos, enemies):
        penalty = 0
        for _, epos, eradius, ebombs in enemies:
            if ebombs <= 0:
                continue
            if pos in self._blast_tiles(grid, epos[0], epos[1], eradius):
                penalty += 10
        return penalty