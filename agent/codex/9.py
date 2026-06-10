from collections import deque


class Agent:
    """
    ResearchHybridAgent

    A pure-Python, CPU-cheap hybrid agent built around:
      - exact-ish bomb danger projection with chain reactions,
      - action pruning by time-expanded BFS escape search,
      - one-step pessimistic action scoring,
      - farming / item economy early, trap pressure when safe.

    It deliberately does not depend on any opponent implementation.
    """

    team_id = "ResearchHybridV1"

    # Engine action mapping: row/col deltas, despite confusing action names in docs.
    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    MOVE_ACTIONS = [1, 2, 3, 4]

    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.turn = 0
        # Track bomb radius at creation because obs does not expose Bomb.radius.
        self.bomb_radius_memory = {}
        self.last_positions = None

    def act(self, obs: dict) -> int:
        self.turn += 1
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0

        me = players[self.agent_id]
        my_pos = (int(me[0]), int(me[1]))
        bombs_left = int(me[3])
        my_radius = max(1, min(5, int(me[4]) + 1))

        self._refresh_bomb_memory(bombs, players)

        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        alive_enemy_ids = [i for i, p in enumerate(players) if i != self.agent_id and int(p[2]) == 1]
        enemies = [(int(players[i][0]), int(players[i][1])) for i in alive_enemy_ids]

        # Movement passability in the real engine allows player overlap; only walls/boxes/old bombs block.
        blocked_base = set(bomb_positions)
        blocked_base.discard(my_pos)

        danger, det_times = self._danger_schedule(grid, bombs, players, horizon=10)
        immediate = danger.get(1, set())
        danger_union = set().union(*(danger.get(t, set()) for t in range(1, 8))) if danger else set()

        valid = self._valid_root_actions(grid, my_pos, blocked_base)

        # Hard safety shield. If in immediate danger, prioritize time-expanded escape only.
        if my_pos in immediate:
            escape = self._best_escape_action(grid, my_pos, blocked_base, danger, horizon=10)
            if escape is not None:
                return escape
            # Last resort: any move not exploding this tick.
            safe_now = [a for a in valid if self._next(my_pos, a) not in immediate]
            return safe_now[0] if safe_now else 0


        # Score movement/stay actions under the existing danger model.
        best_action = 0
        best_score = -10**18
        for action in valid:
            score = self._score_action(
                grid, players, bombs, my_pos, action, blocked_base, danger,
                enemies, alive_enemy_ids, bombs_left, my_radius, hypothetical_bomb=False,
            )
            if score > best_score:
                best_score = score
                best_action = action

        # Score bomb placement separately with augmented danger model and escape proof.
        if bombs_left > 0 and my_pos not in bomb_positions:
            hyp_bombs = self._with_hypothetical_bomb(bombs, my_pos, self.agent_id, timer=7, radius=my_radius)
            hyp_danger, _ = self._danger_schedule(grid, hyp_bombs, players, horizon=10)
            if my_pos not in hyp_danger.get(1, set()):
                shifted_hyp = self._shift_danger(hyp_danger, 1)
                can_escape = self._survives_from(grid, my_pos, blocked_base | {my_pos}, shifted_hyp, horizon=9, allow_start_bomb=True)
                if can_escape:
                    # Pessimistic anti-trap filter: a bomb is allowed only if the post-bomb
                    # state has enough escape fanout, unless it is a meaningful direct attack.
                    post_safe, post_frontier = self._reachable_safe_stats(grid, my_pos, blocked_base | {my_pos}, shifted_hyp, depth=7)
                    boxes_here = self._count_boxes_in_blast(grid, my_pos, my_radius)
                    blast_here = self._blast_tiles(grid, my_pos[0], my_pos[1], my_radius)
                    enemy_hit = any(ep in blast_here for ep in enemies)
                    allow_bomb = (post_safe >= 8 or post_frontier >= 2 or enemy_hit or boxes_here >= 2)
                    if allow_bomb:
                        bomb_score = self._score_action(
                            grid, players, hyp_bombs, my_pos, 5, blocked_base | {my_pos}, hyp_danger,
                            enemies, alive_enemy_ids, bombs_left, my_radius, hypothetical_bomb=True,
                        )
                        # Add direct tactical value of the newly placed bomb.
                        bomb_score += self._bomb_value(grid, my_pos, my_radius, enemies, players, hyp_bombs, hyp_danger)
                        if bomb_score > best_score:
                            best_score = bomb_score
                            best_action = 5


        return int(best_action)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score_action(self, grid, players, bombs, my_pos, action, blocked, danger,
                      enemies, enemy_ids, bombs_left, my_radius, hypothetical_bomb=False):
        if action == 5:
            npos = my_pos
        else:
            npos = self._next(my_pos, action)
            if action != 0 and (not self._passable(grid, *npos) or npos in blocked):
                return -10**12

        if npos in danger.get(1, set()):
            return -10**11

        bomb_cells_now = {(int(b[0]), int(b[1])) for b in bombs}
        if action == 0 and npos in bomb_cells_now:
            return -10**9

        # After the chosen root action, danger[1] has already been checked.
        # Future planning must therefore be relative to the next decision point.
        future_danger = self._shift_danger(danger, 1)

        # Time-expanded survivability is the action filter. It prunes self-traps.
        future_blocked = set(blocked)
        if hypothetical_bomb:
            # Standing on the new bomb is legal at t=0 only; the BFS helper handles this.
            survives = self._survives_from(grid, npos, future_blocked, future_danger, horizon=9, allow_start_bomb=True)
        else:
            survives = self._survives_from(grid, npos, future_blocked, future_danger, horizon=9)
        if not survives:
            return -10**10

        score = 0.0

        # Prefer states with a large reachable safe component. This is the pessimistic robustness term.
        safe_count, frontier_count = self._reachable_safe_stats(grid, npos, future_blocked, future_danger, depth=8)
        score += 18.0 * safe_count + 8.0 * frontier_count

        min_danger = self._min_danger_time(npos, future_danger, horizon=9)
        score += 12.0 * min(min_danger, 10)
        if min_danger <= 2:
            score -= 320.0
        elif min_danger <= 4:
            score -= 140.0

        # Center control matters because late auto-spawn and routes are richer near the center.
        h, w = grid.shape
        cx, cy = (h - 1) / 2.0, (w - 1) / 2.0
        score -= 1.7 * (abs(npos[0] - cx) + abs(npos[1] - cy))

        # Item economy: capacity is valuable when low; radius becomes valuable after capacity safety.
        cap = int(players[self.agent_id][3])
        bonus = int(players[self.agent_id][4])
        item_score = self._nearest_item_score(grid, npos, future_blocked, future_danger, cap, bonus)
        score += item_score
        if grid[npos[0], npos[1]] == self.ITEM_CAPACITY:
            score += 260.0
        elif grid[npos[0], npos[1]] == self.ITEM_RADIUS:
            score += 220.0

        # Farm pressure and tie-break stats: move toward high-value bombing tiles.
        farm_score = self._nearest_bomb_spot_score(grid, npos, future_blocked, future_danger, my_radius)
        score += farm_score

        # Enemy pressure. Early game: do not overchase; after some scaling, pressure harder.
        if enemies:
            d_enemy = self._nearest_distance(grid, npos, set(enemies), future_blocked, future_danger, max_depth=12)
            if d_enemy is not None:
                power = int(players[self.agent_id][4]) + int(players[self.agent_id][3])
                chase_weight = 5.0 + min(18.0, 3.0 * power) + (4.0 if self.turn > 180 else 0.0)
                score += max(0.0, 14.0 - d_enemy) * chase_weight

            # Avoid being too close to a stronger opponent unless we have an escape fanout.
            for eid, ep in zip(enemy_ids, enemies):
                md = abs(ep[0] - npos[0]) + abs(ep[1] - npos[1])
                enemy_power = int(players[eid][3]) + int(players[eid][4])
                my_power = int(players[self.agent_id][3]) + int(players[self.agent_id][4])
                if md <= 2 and enemy_power >= my_power and safe_count < 9:
                    score -= 120.0 / max(1, md)

        # Stop is okay only when tactically useful; otherwise add mild mobility preference.
        if action == 0:
            score -= 6.0
        else:
            score += 4.0

        # If we would walk into a cell an enemy can bomb immediately, require more safe fanout.
        if self._enemy_bomb_threat(grid, npos, players, enemy_ids) and safe_count < 12:
            score -= 110.0

        return score

    def _bomb_value(self, grid, pos, radius, enemies, players, bombs, danger):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        boxes = sum(1 for t in blast if grid[t[0], t[1]] == self.BOX)
        value = 0.0
        # Bombing has intrinsic tie-break value, but avoid spam by keeping it modest.
        value += 30.0
        value += 115.0 * boxes
        if boxes >= 2:
            value += 75.0

        # Direct line hits and trap potential.
        for ep in enemies:
            if ep in blast:
                value += 520.0
                # Estimate whether enemy has an escape from our blast in 7 plies.
                enemy_blocked = {(int(b[0]), int(b[1])) for b in bombs}
                enemy_blocked.discard(ep)
                can_enemy_escape = self._survives_from(grid, ep, enemy_blocked, danger, horizon=7)
                if not can_enemy_escape:
                    value += 900.0
            else:
                md = abs(ep[0] - pos[0]) + abs(ep[1] - pos[1])
                if md <= radius + 1:
                    value += max(0.0, 160.0 - 25.0 * md)

        # Bombing inside a tiny cul-de-sac is riskier even if escape exists.
        degree = self._free_degree(grid, pos, {(int(b[0]), int(b[1])) for b in bombs})
        if degree <= 1:
            value -= 80.0
        return value

    # ------------------------------------------------------------------
    # Danger and bomb model
    # ------------------------------------------------------------------

    def _refresh_bomb_memory(self, bombs, players):
        live = set()
        for b in bombs:
            bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            key = (bx, by, owner)
            live.add(key)
            if key not in self.bomb_radius_memory or timer >= self.bomb_radius_memory[key][0]:
                radius = 2
                if 0 <= owner < len(players):
                    radius = max(1, min(5, int(players[owner][4]) + 1))
                self.bomb_radius_memory[key] = (timer, radius)
            else:
                old_timer, radius = self.bomb_radius_memory[key]
                self.bomb_radius_memory[key] = (timer, radius)
        # Drop disappeared bombs.
        for key in list(self.bomb_radius_memory.keys()):
            if key not in live:
                self.bomb_radius_memory.pop(key, None)

    def _bomb_infos(self, bombs, players):
        infos = []
        for b in bombs:
            bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            radius = None
            key = (bx, by, owner)
            if key in self.bomb_radius_memory:
                radius = self.bomb_radius_memory[key][1]
            if radius is None:
                radius = 2
                if 0 <= owner < len(players):
                    radius = max(1, min(5, int(players[owner][4]) + 1))
            infos.append({"pos": (bx, by), "timer": max(1, timer), "owner": owner, "radius": radius})
        return infos

    def _with_hypothetical_bomb(self, bombs, pos, owner, timer, radius):
        # Keep a list-like tuple format compatible with our helpers.
        out = []
        for b in bombs:
            out.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        out.append((int(pos[0]), int(pos[1]), int(timer), int(owner)))
        # Preload memory so _bomb_infos uses the hypothetical radius.
        self.bomb_radius_memory[(int(pos[0]), int(pos[1]), int(owner))] = (int(timer), int(radius))
        return out

    def _danger_schedule(self, grid, bombs, players, horizon=10):
        infos = self._bomb_infos(bombs, players)
        n = len(infos)
        det = [min(horizon + 1, max(1, info["timer"])) for info in infos]

        changed = True
        while changed:
            changed = False
            order = sorted(range(n), key=lambda i: det[i])
            for i in order:
                if det[i] > horizon:
                    continue
                blast = self._blast_tiles(grid, infos[i]["pos"][0], infos[i]["pos"][1], infos[i]["radius"])
                for j in range(n):
                    if i == j:
                        continue
                    if det[j] > det[i] and infos[j]["pos"] in blast:
                        det[j] = det[i]
                        changed = True

        danger = {t: set() for t in range(1, horizon + 1)}
        for i, info in enumerate(infos):
            t = det[i]
            if 1 <= t <= horizon:
                danger[t].update(self._blast_tiles(grid, info["pos"][0], info["pos"][1], info["radius"]))
        return danger, det

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(int(bx), int(by))}
        for dx, dy in self.DIRS:
            for r in range(1, int(radius) + 1):
                x, y = int(bx) + dx * r, int(by) + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == self.WALL:
                    break
                tiles.add((x, y))
                if cell == self.BOX:
                    break
        return tiles

    # ------------------------------------------------------------------
    # Search / pathfinding helpers
    # ------------------------------------------------------------------

    def _valid_root_actions(self, grid, pos, blocked):
        actions = [0]
        for a in self.MOVE_ACTIONS:
            npos = self._next(pos, a)
            if self._passable(grid, *npos) and npos not in blocked:
                actions.append(a)
        return actions

    def _best_escape_action(self, grid, pos, blocked, danger, horizon=10):
        candidates = self._valid_root_actions(grid, pos, blocked)
        best = None
        best_score = -10**9
        for a in candidates:
            npos = self._next(pos, a)
            if npos in danger.get(1, set()):
                continue
            future_danger = self._shift_danger(danger, 1)
            if not self._survives_from(grid, npos, blocked, future_danger, horizon=max(1, horizon - 1)):
                continue
            safe_count, frontier = self._reachable_safe_stats(grid, npos, blocked, future_danger, depth=8)
            min_d = self._min_danger_time(npos, future_danger, max(1, horizon - 1))
            score = 1000 + 25 * safe_count + 35 * min_d + 5 * frontier
            if a != 0:
                score += 4
            if score > best_score:
                best_score = score
                best = a
        return best

    def _survives_from(self, grid, start, blocked, danger, horizon=10, allow_start_bomb=False):
        """Time-expanded BFS: exists a sequence surviving every explosion tick through horizon."""
        if start in danger.get(0, set()):
            return False
        q = deque([(start, 0)])
        seen = {(start, 0)}
        best_t = 0
        while q:
            pos, t = q.popleft()
            best_t = max(best_t, t)
            if t >= horizon:
                return True
            # If this state has high future margin and some fanout, accept early.
            if t >= 3 and self._min_danger_time(pos, danger, horizon) > horizon:
                if self._free_degree(grid, pos, blocked) >= 2:
                    return True
            for a in [0, 1, 2, 3, 4]:
                npos = self._next(pos, a)
                nt = t + 1
                if a != 0:
                    if not self._passable(grid, *npos):
                        continue
                    if npos in blocked:
                        # The starting bomb cell is legal only at t=0 while standing on it, not as a destination.
                        continue
                else:
                    # Standing on a bomb is legal only at the start; if blocked later, do not wait there.
                    if pos in blocked and not (allow_start_bomb and pos == start and t == 0):
                        continue
                if npos in danger.get(nt, set()):
                    continue
                state = (npos, nt)
                if state in seen:
                    continue
                seen.add(state)
                q.append(state)
        return best_t >= horizon

    def _reachable_safe_stats(self, grid, start, blocked, danger, depth=8):
        q = deque([(start, 0)])
        seen = {(start, 0)}
        safe_positions = set()
        frontier = 0
        while q:
            pos, d = q.popleft()
            if pos not in danger.get(d, set()):
                # Count cells that are not scheduled to explode soon after arrival.
                if self._min_danger_time(pos, danger, depth + 2) > d + 1:
                    safe_positions.add(pos)
            if d >= depth:
                frontier += 1
                continue
            for a in [0, 1, 2, 3, 4]:
                npos = self._next(pos, a)
                nd = d + 1
                if a != 0:
                    if not self._passable(grid, *npos) or npos in blocked:
                        continue
                if npos in danger.get(nd, set()):
                    continue
                st = (npos, nd)
                if st not in seen:
                    seen.add(st)
                    q.append(st)
        return len(safe_positions), frontier

    def _nearest_distance(self, grid, start, targets, blocked, danger=None, max_depth=99):
        if not targets:
            return None
        if start in targets:
            return 0
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if d >= max_depth:
                continue
            for a in self.MOVE_ACTIONS:
                npos = self._next(pos, a)
                nd = d + 1
                if npos in seen:
                    continue
                if not self._passable(grid, *npos) or npos in blocked:
                    continue
                if danger is not None and npos in danger.get(nd, set()):
                    continue
                if npos in targets:
                    return nd
                seen.add(npos)
                q.append((npos, nd))
        return None

    def _first_step_to_targets(self, grid, start, targets, blocked, danger=None, max_depth=99):
        if not targets:
            return None
        q = deque([(start, 0, None)])
        seen = {start}
        while q:
            pos, d, first = q.popleft()
            if d > 0 and pos in targets:
                return first
            if d >= max_depth:
                continue
            # Prefer deterministic direct actions over STOP in pathing.
            for a in self.MOVE_ACTIONS:
                npos = self._next(pos, a)
                nd = d + 1
                if npos in seen:
                    continue
                if not self._passable(grid, *npos) or npos in blocked:
                    continue
                if danger is not None and npos in danger.get(nd, set()):
                    continue
                seen.add(npos)
                q.append((npos, nd, a if first is None else first))
        return None


    # ------------------------------------------------------------------
    # Tactical evaluation helpers
    # ------------------------------------------------------------------

    def _nearest_item_score(self, grid, pos, blocked, danger, bombs_left, bomb_bonus):
        targets = []
        h, w = grid.shape
        for x in range(h):
            for y in range(w):
                cell = int(grid[x, y])
                if cell == self.ITEM_CAPACITY or cell == self.ITEM_RADIUS:
                    targets.append((x, y, cell))
        if not targets:
            return 0.0
        best = 0.0
        for x, y, cell in targets:
            d = self._nearest_distance(grid, pos, {(x, y)}, blocked, danger, max_depth=14)
            if d is None:
                continue
            base = 340.0 if cell == self.ITEM_CAPACITY else 300.0
            if cell == self.ITEM_CAPACITY and bombs_left <= 1:
                base += 130.0
            if cell == self.ITEM_RADIUS and bomb_bonus <= 1:
                base += 90.0
            best = max(best, base / (d + 1))
        return best

    def _nearest_bomb_spot_score(self, grid, pos, blocked, danger, radius):
        # Candidate tiles where placing a bomb destroys at least one box. Score by boxes hit / distance.
        best = 0.0
        h, w = grid.shape
        # Keep it small: all passable cells on 13x13 is fine.
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if not self._passable(grid, x, y) or (x, y) in blocked:
                    continue
                boxes = self._count_boxes_in_blast(grid, (x, y), radius)
                if boxes <= 0:
                    continue
                d = self._nearest_distance(grid, pos, {(x, y)}, blocked, danger, max_depth=14)
                if d is None:
                    continue
                val = (135.0 * boxes + (60.0 if boxes >= 2 else 0.0)) / (d + 1)
                if d == 0:
                    val += 30.0 * boxes
                best = max(best, val)
        return best

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for x, y in self._blast_tiles(grid, pos[0], pos[1], radius) if int(grid[x, y]) == self.BOX)

    def _enemy_bomb_threat(self, grid, pos, players, enemy_ids):
        # Cell is in immediate line of a bomb that any enemy standing now could place.
        for eid in enemy_ids:
            ep = (int(players[eid][0]), int(players[eid][1]))
            if int(players[eid][3]) <= 0:
                continue
            er = max(1, min(5, int(players[eid][4]) + 1))
            if pos in self._blast_tiles(grid, ep[0], ep[1], er):
                return True
        return False


    def _shift_danger(self, danger, offset):
        if offset <= 0:
            return danger
        max_t = max(danger.keys()) if danger else 0
        shifted = {}
        for t in range(1, max(1, max_t - offset) + 1):
            shifted[t] = set(danger.get(t + offset, set()))
        return shifted

    # ------------------------------------------------------------------
    # Primitive helpers
    # ------------------------------------------------------------------

    def _next(self, pos, action):
        dx, dy = self.MOVES.get(int(action), (0, 0))
        return (int(pos[0]) + dx, int(pos[1]) + dy)

    def _in_bounds(self, grid, x, y):
        return 0 <= int(x) < grid.shape[0] and 0 <= int(y) < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and int(grid[int(x), int(y)]) in (self.GRASS, self.ITEM_RADIUS, self.ITEM_CAPACITY)

    def _min_danger_time(self, pos, danger, horizon=10):
        for t in range(1, horizon + 1):
            if pos in danger.get(t, set()):
                return t
        return horizon + 1

    def _free_degree(self, grid, pos, blocked):
        deg = 0
        for a in self.MOVE_ACTIONS:
            npos = self._next(pos, a)
            if self._passable(grid, *npos) and npos not in blocked:
                deg += 1
        return deg
