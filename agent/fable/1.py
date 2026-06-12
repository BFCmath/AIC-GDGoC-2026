from collections import deque


class Agent:
    """
    FableStatRaceV1

    Hybrid search agent built around three pillars derived from match analysis:

    1. Survival core: chain-aware bomb danger schedule + time-expanded escape
       BFS + pessimistic anti-trap shields (the strongest known safety model).

    2. Exact stat-race tracking: nearly all matches between competent agents
       truncate at 500 steps, where rank = (kills, boxes, items, bombs).
       This agent reconstructs every player's tie-break stats from observation
       deltas and plays the tie-break race explicitly:
         - strictly ahead late  -> lock in the win, play maximally safe,
         - behind               -> push the exact stat that flips the rank
                                   (one kill beats any box lead).

    3. Kill harvesting: guaranteed-kill bombs (enemy time-expanded escape
       provably fails), trap-setup approach bonuses, and blast saturation on
       enemies that demonstrably linger in danger (behavioral vulnerability,
       not opponent fingerprinting).
    """

    team_id = "FableStatRaceV1"

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
        self.bomb_radius_memory = {}
        # Stat-race tracking state
        self.prev_snapshot = None
        self.tracked_stats = [
            {"kills": 0, "boxes": 0, "items": 0, "bombs": 0} for _ in range(4)
        ]
        # EWMA of time each player spends inside scheduled blast tiles.
        self.exposure = [0.0] * 4
        # Consecutive turns each player has held the same position (camping).
        self.stuck = [0] * 4
        self._last_pos = [None] * 4

    # ==================================================================
    # Main decision loop
    # ==================================================================

    def act(self, obs: dict) -> int:
        self.turn += 1
        self._blast_cache = {}
        self._hyp_trap_cache = {}
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            self._snapshot(grid, players, bombs)
            return 0

        # Update stat tracker BEFORE bomb memory refresh (it needs radii of
        # bombs that just exploded, which refresh would drop).
        self._update_tracker(grid, players, bombs)

        me = players[self.agent_id]
        my_pos = (int(me[0]), int(me[1]))
        bombs_left = int(me[3])
        my_radius = max(1, min(5, int(me[4]) + 1))

        self._refresh_bomb_memory(bombs, players)

        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        alive_enemy_ids = [i for i, p in enumerate(players) if i != self.agent_id and int(p[2]) == 1]
        enemies = [(int(players[i][0]), int(players[i][1])) for i in alive_enemy_ids]

        blocked_all = set(bomb_positions)
        blocked_base = set(bomb_positions)
        blocked_base.discard(my_pos)

        danger, det_times = self._danger_schedule(grid, bombs, players, horizon=10)
        immediate = danger.get(1, set())

        # Exposure bookkeeping (behavioral vulnerability signal). Only standing
        # in a cell about to explode counts: competent agents clear blast zones
        # early, so their exposure stays near zero, while clumsy agents linger.
        imminent = danger.get(1, set()) | danger.get(2, set())
        for i in range(4):
            if i != self.agent_id and int(players[i][2]) == 1:
                pos_i = (int(players[i][0]), int(players[i][1]))
                self.exposure[i] = 0.96 * self.exposure[i] + (1.0 if pos_i in imminent else 0.0)
                if pos_i == self._last_pos[i]:
                    self.stuck[i] += 1
                else:
                    self.stuck[i] = 0
                self._last_pos[i] = pos_i

        # Race mode for this turn (sets self._mode fields used in scoring).
        self._compute_race_mode(grid, players, alive_enemy_ids)

        # Joint pessimistic danger: ALL nearby armed enemies bomb this tick.
        # Catches two-enemy corridor seals that per-enemy models miss.
        self._joint_trap = None
        near_armed = []
        for eid, ep in zip(alive_enemy_ids, enemies):
            if (
                int(players[eid][3]) > 0
                and ep not in bomb_positions
                and abs(ep[0] - my_pos[0]) + abs(ep[1] - my_pos[1]) <= 6
            ):
                near_armed.append((eid, ep))
        if len(near_armed) >= 2:
            jb = [tuple(int(v) for v in b) for b in bombs]
            keys = []
            for eid, ep in near_armed:
                er = max(1, min(5, int(players[eid][4]) + 1))
                jb = self._with_hypothetical_bomb(jb, ep, eid, timer=7, radius=er)
                keys.append((ep[0], ep[1], eid))
            jd, _ = self._danger_schedule(grid, jb, players, horizon=10)
            for k in keys:
                self.bomb_radius_memory.pop(k, None)
            self._joint_trap = (
                self._shift_danger(jd, 1),
                set(blocked_all) | {ep for _, ep in near_armed},
            )

        valid = self._valid_root_actions(grid, my_pos, blocked_base)

        # ---- Hard safety shield ----
        if my_pos in immediate:
            escape = self._best_escape_action(grid, my_pos, blocked_all, danger, horizon=10)
            if escape is not None:
                self._snapshot(grid, players, bombs)
                return escape
            safe_now = [a for a in valid if self._next(my_pos, a) not in immediate]
            self._snapshot(grid, players, bombs)
            return safe_now[0] if safe_now else 0

        danger_eta = self._min_danger_time(my_pos, danger, horizon=10)
        if danger_eta <= 3 or (danger_eta <= 4 and self._enemy_bomb_threat(grid, my_pos, players, alive_enemy_ids)):
            escape = self._best_escape_action(grid, my_pos, blocked_all, danger, horizon=10)
            if escape is not None and escape != 0:
                self._snapshot(grid, players, bombs)
                return escape

        # ---- Guaranteed-kill bomb (kills are the first tie-breaker) ----
        if bombs_left > 0 and my_pos not in bomb_positions and enemies:
            kill_action = self._guaranteed_kill_bomb(
                grid, players, bombs, my_pos, my_radius, blocked_all, alive_enemy_ids, enemies
            )
            if kill_action is not None:
                self._snapshot(grid, players, bombs)
                return kill_action

        # ---- Opportunistic adjacent item pickup ----
        item_moves = []
        for action in valid:
            if action == 0:
                continue
            npos = self._next(my_pos, action)
            if npos not in danger.get(1, set()) and int(grid[npos[0], npos[1]]) in (self.ITEM_CAPACITY, self.ITEM_RADIUS):
                future_danger = self._shift_danger(danger, 1)
                if self._survives_from(grid, npos, blocked_all, future_danger, horizon=8):
                    item_moves.append((int(grid[npos[0], npos[1]]) == self.ITEM_CAPACITY, action))
        if item_moves:
            item_moves.sort(reverse=True)
            self._snapshot(grid, players, bombs)
            return int(item_moves[0][1])

        # ---- Near item chase ----
        item_targets = set()
        h, w = grid.shape
        for ix in range(h):
            for iy in range(w):
                if int(grid[ix, iy]) in (self.ITEM_CAPACITY, self.ITEM_RADIUS):
                    item_targets.add((ix, iy))
        if item_targets:
            boxes_now = self._count_boxes_in_blast(grid, my_pos, my_radius) if bombs_left > 0 and my_pos not in bomb_positions else 0
            dist_map = self._distance_map(grid, my_pos, blocked_all, danger, max_depth=7)
            best = None
            best_score = -1.0
            for ip in item_targets:
                d = dist_map.get(ip)
                if d is None or d <= 0:
                    continue
                # Skip items a competent enemy will reach clearly first (wasted
                # travel). Items near clumsy enemies stay attractive: they are
                # both collectable and bait toward huntable targets.
                contested = [
                    abs(players[eid][0] - ip[0]) + abs(players[eid][1] - ip[1])
                    for eid in alive_enemy_ids
                    if not self._huntable(eid)
                ]
                if contested and min(contested) < d - 1:
                    continue
                cell = int(grid[ip[0], ip[1]])
                base = 1000.0 if cell == self.ITEM_CAPACITY else 820.0
                val = base / (d + 1)
                if val > best_score:
                    best_score, best = val, (ip, d)
            if best is not None:
                ip, d = best
                chase_d = 6 if self._push_items else 2
                if (
                    d <= chase_d
                    or (self.turn > 155 and boxes_now == 0 and d <= 6)
                    or (self.turn > 230 and boxes_now <= 1 and d <= 5)
                ):
                    step = self._first_step_to_targets(grid, my_pos, {ip}, blocked_all, danger, max_depth=7)
                    if step is not None:
                        self._snapshot(grid, players, bombs)
                        return int(step)

        # ---- Score movement/stay actions ----
        best_action = 0
        best_score = -10**18
        for action in valid:
            score = self._score_action(
                grid, players, bombs, my_pos, action, blocked_all, danger,
                enemies, alive_enemy_ids, bombs_left, my_radius, hypothetical_bomb=False,
            )
            if score > best_score:
                best_score = score
                best_action = action

        # ---- Score bomb placement ----
        if bombs_left > 0 and my_pos not in bomb_positions:
            hyp_bombs = self._with_hypothetical_bomb(bombs, my_pos, self.agent_id, timer=7, radius=my_radius)
            hyp_danger, hyp_det = self._danger_schedule(grid, hyp_bombs, players, horizon=10)
            my_det = hyp_det[-1] if hyp_det else 7
            if my_pos not in hyp_danger.get(1, set()):
                shifted_hyp = self._shift_danger(hyp_danger, 1)
                can_escape = self._survives_from(grid, my_pos, blocked_all | {my_pos}, shifted_hyp, horizon=9, allow_start_bomb=True)
                if can_escape and enemies:
                    can_escape = self._escape_robust_to_simul_bombs(
                        grid, players, my_pos, hyp_bombs, blocked_all, alive_enemy_ids, enemies
                    )
                if can_escape:
                    post_safe, post_frontier = self._reachable_safe_stats(grid, my_pos, blocked_all | {my_pos}, shifted_hyp, depth=7)
                    blast_here = self._blast_tiles(grid, my_pos[0], my_pos[1], my_radius)
                    # Credit-timing: a box already covered by an earlier enemy
                    # blast is lost (their bomb pops it before mine), but a box
                    # popping the SAME tick as mine is shared credit (chain
                    # poaching) and counts in full.
                    boxes_here = 0
                    for bt in blast_here:
                        if int(grid[bt[0], bt[1]]) == self.BOX:
                            destroyed_at = self._min_danger_time(bt, danger, horizon=10)
                            if my_det <= destroyed_at:
                                boxes_here += 1
                    enemy_hit = any(ep in blast_here for ep in enemies)
                    allow_bomb = (
                        post_safe >= 7
                        or post_frontier >= 2
                        or enemy_hit
                        or (boxes_here >= 1 and post_safe >= 4)
                        or (self.turn > 220 and post_safe >= 18 and post_frontier >= 2)
                    )
                    if self._safe_mode:
                        # Leader lock-in: only ultra-safe bombs are allowed.
                        allow_bomb = allow_bomb and post_safe >= 14 and post_frontier >= 2 and not self._enemy_bomb_threat(grid, my_pos, players, alive_enemy_ids)
                    if allow_bomb:
                        # Greedy safe farming, but defer a 1-box bomb when a much
                        # richer spot is within two steps (box race efficiency).
                        if boxes_here >= 1 and post_safe >= 3 and not self._safe_mode:
                            defer = False
                            if boxes_here == 1 and bombs_left == 1:
                                rich = self._best_spot_within(grid, my_pos, blocked_all, danger, my_radius, depth=2)
                                if rich >= 3:
                                    defer = True
                            if not defer:
                                self._snapshot(grid, players, bombs)
                                return 5
                        bomb_score = self._score_action(
                            grid, players, hyp_bombs, my_pos, 5, blocked_all | {my_pos}, hyp_danger,
                            enemies, alive_enemy_ids, bombs_left, my_radius, hypothetical_bomb=True,
                        )
                        bomb_score += self._bomb_value(grid, my_pos, my_radius, enemies, players, hyp_bombs, hyp_danger, alive_enemy_ids, eff_boxes=boxes_here)
                        # Kill-share: an enemy already doomed by existing bombs
                        # still credits OUR kill stat if our blast covers them on
                        # the same detonation tick (chain alignment).
                        for eid, ep in zip(alive_enemy_ids, enemies):
                            if ep in blast_here:
                                eta_e = self._min_danger_time(ep, danger, horizon=8)
                                if eta_e <= 6:
                                    e_blocked = set(bomb_positions)
                                    e_blocked.discard(ep)
                                    if my_det <= eta_e + 1 and not self._survives_from(grid, ep, e_blocked, danger, horizon=7):
                                        bomb_score += 800.0
                        if boxes_here > 0:
                            bomb_score += 260.0 + 165.0 * boxes_here
                            if boxes_here == 1 and post_safe >= 5:
                                bomb_score += 90.0
                        elif self.turn > 220 and post_safe >= 18 and post_frontier >= 2:
                            bomb_score += 95.0
                            if self.turn > 320:
                                bomb_score += 70.0
                        if self.turn > 260 and post_safe >= 17 and post_frontier >= 2:
                            bomb_score += 35.0
                        if self.turn > 350 and post_safe >= 15:
                            bomb_score += 20.0
                        if self._push_bombs and post_safe >= 12 and post_frontier >= 2:
                            bomb_score += 120.0
                        if self._safe_mode:
                            bomb_score -= 150.0
                        if bomb_score > best_score:
                            best_score = bomb_score
                            best_action = 5

        self._snapshot(grid, players, bombs)
        return int(best_action)

    # ==================================================================
    # Stat-race tracking (exact tie-break bookkeeping from obs deltas)
    # ==================================================================

    def _snapshot(self, grid, players, bombs):
        self.prev_snapshot = (
            grid.copy(),
            [[int(v) for v in p] for p in players],
            [[int(v) for v in b] for b in bombs],
            [int(p[2]) == 1 for p in players],
        )

    def _update_tracker(self, grid, players, bombs):
        if self.prev_snapshot is None:
            return
        pgrid, pplayers, pbombs, palive = self.prev_snapshot
        cur_keys = {(int(b[0]), int(b[1]), int(b[3])) for b in bombs}
        prev_keys = {(b[0], b[1], b[3]) for b in pbombs}

        # Bombs placed this step.
        for k in cur_keys - prev_keys:
            if 0 <= k[2] < 4:
                self.tracked_stats[k[2]]["bombs"] += 1

        # Exploded bombs -> attribute boxes and kills.
        exploded = prev_keys - cur_keys
        affected = {}
        for (bx, by, owner) in exploded:
            mem = self.bomb_radius_memory.get((bx, by, owner))
            if mem is not None:
                radius = mem[1]
            else:
                radius = 2
                if 0 <= owner < 4:
                    radius = max(1, min(5, pplayers[owner][4] + 1))
            for t in self._blast_tiles_nocache(pgrid, bx, by, radius):
                affected.setdefault(t, set()).add(owner)

        for t, owners in affected.items():
            if int(pgrid[t[0], t[1]]) == self.BOX:
                for o in owners:
                    if 0 <= o < 4:
                        self.tracked_stats[o]["boxes"] += 1

        # Items: pickup resolves after movement, before explosions. A tile that
        # held an item last step and has exactly one mover on it now was collected.
        for i in range(4):
            if palive[i]:
                px, py = int(players[i][0]), int(players[i][1])
                if int(pgrid[px, py]) in (self.ITEM_RADIUS, self.ITEM_CAPACITY):
                    occ = sum(
                        1 for j in range(4)
                        if palive[j] and int(players[j][0]) == px and int(players[j][1]) == py
                    )
                    if occ == 1:
                        self.tracked_stats[i]["items"] += 1

        # Kills: players alive last step, dead now, standing on a blast tile.
        for i in range(4):
            if palive[i] and int(players[i][2]) != 1:
                t = (int(players[i][0]), int(players[i][1]))
                for o in affected.get(t, set()):
                    if o != i and 0 <= o < 4:
                        self.tracked_stats[o]["kills"] += 1

    def _tb_key(self, i):
        s = self.tracked_stats[i]
        return (s["kills"], s["boxes"], s["items"], s["bombs"])

    def _compute_race_mode(self, grid, players, alive_enemy_ids):
        self._safe_mode = False
        self._push_kills = False
        self._push_boxes = False
        self._push_items = False
        self._push_bombs = False
        self._hunt_target = None

        if not alive_enemy_ids:
            return

        # The opening is a pure box/economy sprint (the map's boxes are gone by
        # ~step 100); stat-race steering only matters once farming winds down.
        if self.turn < 120:
            return

        my_key = self._tb_key(self.agent_id)
        opp_keys = [(self._tb_key(i), i) for i in alive_enemy_ids]
        best_key, best_id = max(opp_keys)

        boxes_left = 0
        h, w = grid.shape
        for x in range(h):
            for y in range(w):
                if int(grid[x, y]) == self.BOX:
                    boxes_left += 1
        self._boxes_left = boxes_left

        if my_key > best_key:
            # Margin at the first differing stat.
            margin = 0
            stat_idx = 3
            for i in range(4):
                if my_key[i] != best_key[i]:
                    margin = my_key[i] - best_key[i]
                    stat_idx = i
                    break
            # Lock in the win late only when the lead is effectively uncatchable:
            # kills cannot be farmed back; a box lead is safe once remaining
            # boxes are fewer than the margin; an item lead needs a buffer
            # because items keep auto-spawning.
            locked = (
                stat_idx == 0
                or (stat_idx == 1 and margin > boxes_left)
                or (stat_idx == 2 and margin >= 4)
            )
            if self.turn > 380 and locked:
                self._safe_mode = True
            if stat_idx == 3 or (stat_idx == 2 and margin <= 1):
                self._push_bombs = True
                self._push_items = True
        elif my_key == best_key:
            self._push_items = True
            self._push_bombs = True
        else:
            for i in range(4):
                if my_key[i] < best_key[i]:
                    if i == 0:
                        self._push_kills = True
                    elif i == 1:
                        # A kill flips any box deficit; if boxes are running out
                        # or the deficit is large, hunting is the better path.
                        deficit = best_key[1] - my_key[1]
                        if boxes_left <= deficit or (self.turn > 360 and deficit >= 2):
                            self._push_kills = True
                        else:
                            self._push_boxes = True
                    elif i == 2:
                        self._push_items = True
                    else:
                        self._push_bombs = True
                    break
            if self._push_kills and self.turn > 300:
                self._hunt_target = best_id

    # ==================================================================
    # Kill logic
    # ==================================================================

    def _guaranteed_kill_bomb(self, grid, players, bombs, my_pos, my_radius, blocked_all, enemy_ids, enemies):
        blast_here = self._blast_tiles(grid, my_pos[0], my_pos[1], my_radius)
        in_blast = [(eid, ep) for eid, ep in zip(enemy_ids, enemies) if ep in blast_here]
        if not in_blast:
            return None
        hyp_bombs = self._with_hypothetical_bomb(bombs, my_pos, self.agent_id, timer=7, radius=my_radius)
        hyp_danger, _ = self._danger_schedule(grid, hyp_bombs, players, horizon=10)
        if my_pos in hyp_danger.get(1, set()):
            return None
        shifted = self._shift_danger(hyp_danger, 1)
        if not self._survives_from(grid, my_pos, blocked_all | {my_pos}, shifted, horizon=9, allow_start_bomb=True):
            return None
        if not self._escape_robust_to_simul_bombs(grid, players, my_pos, hyp_bombs, blocked_all, enemy_ids, enemies):
            return None
        bomb_cells = {(int(b[0]), int(b[1])) for b in hyp_bombs}
        for eid, ep in in_blast:
            enemy_blocked = set(bomb_cells)
            allow_start = ep in enemy_blocked
            if not self._survives_from(grid, ep, enemy_blocked, shifted, horizon=8, allow_start_bomb=allow_start):
                # Against a careful (low-exposure) enemy, require real escape
                # fanout for ourselves before committing to the kill bomb.
                if not self._huntable(eid):
                    post_safe, _ = self._reachable_safe_stats(grid, my_pos, blocked_all | {my_pos}, shifted, depth=7)
                    if post_safe < 6:
                        continue
                return 5
        return None

    def _escape_robust_to_simul_bombs(self, grid, players, my_pos, hyp_bombs, blocked_all, enemy_ids, enemies):
        """Pessimistic same-tick model: a nearby armed enemy may place a bomb
        in the SAME step as ours, blocking an escape lane and adding blast.
        Require that our escape still exists under each such joint placement.
        This is the trap that kills greedy farmers in corridors."""
        bomb_cells = {(b[0], b[1]) for b in hyp_bombs} if hyp_bombs is not None else set()
        for eid, ep in zip(enemy_ids, enemies):
            if int(players[eid][3]) <= 0:
                continue
            md = abs(ep[0] - my_pos[0]) + abs(ep[1] - my_pos[1])
            if md > 3:
                continue
            if (int(ep[0]), int(ep[1])) in bomb_cells:
                continue  # enemy stands on a live bomb; cannot place another
            er = max(1, min(5, int(players[eid][4]) + 1))
            joint = self._with_hypothetical_bomb(hyp_bombs, ep, eid, timer=7, radius=er)
            joint_danger, _ = self._danger_schedule(grid, joint, players, horizon=10)
            shifted_joint = self._shift_danger(joint_danger, 1)
            ok = self._survives_from(
                grid, my_pos, blocked_all | {my_pos, ep}, shifted_joint, horizon=9, allow_start_bomb=True
            )
            # Clean up the speculative enemy bomb from radius memory.
            self.bomb_radius_memory.pop((int(ep[0]), int(ep[1]), int(eid)), None)
            if not ok:
                return False
        return True

    def _huntable(self, eid):
        """An enemy is huntable when it demonstrably lingers in danger zones
        (weak/clumsy play), camps in place, or when the race demands
        desperate kill pressure."""
        return self.exposure[eid] > 0.8 or (self._push_kills and self._hunt_target == eid)

    def _vuln(self, eid):
        """Effective vulnerability magnitude used for hunt/saturation weights."""
        return self.exposure[eid]

    def _trap_setup_value(self, grid, players, bombs, npos, my_radius, enemy_ids, enemies, blocked):
        """Bonus for moving to a cell from which a bomb NEXT turn would likely
        seal a nearby enemy. Only evaluated for close enemies to stay cheap."""
        value = 0.0
        for eid, ep in zip(enemy_ids, enemies):
            if not self._huntable(eid):
                continue
            md = abs(ep[0] - npos[0]) + abs(ep[1] - npos[1])
            if md > 3 or md == 0:
                continue
            blast = self._blast_tiles(grid, npos[0], npos[1], my_radius)
            if ep not in blast:
                continue
            # Cheap structural check: enemy escape routes not covered by blast.
            exits = 0
            bomb_cells = {(int(b[0]), int(b[1])) for b in bombs}
            for a in self.MOVE_ACTIONS:
                q = self._next(ep, a)
                if self._passable(grid, *q) and q not in bomb_cells and q not in blast:
                    exits += 1
            if exits == 0:
                value += 420.0 + 90.0 * min(self._vuln(eid), 4.0)
            elif exits == 1:
                value += 130.0 + 45.0 * min(self._vuln(eid), 4.0)
        return value

    # ==================================================================
    # Core scoring
    # ==================================================================

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

        future_danger = self._shift_danger(danger, 1)
        future_blocked = set(blocked) | set(bomb_cells_now)
        if hypothetical_bomb:
            survives = self._survives_from(grid, npos, future_blocked, future_danger, horizon=9, allow_start_bomb=True)
        else:
            survives = self._survives_from(grid, npos, future_blocked, future_danger, horizon=9)
        if not survives:
            return -10**10

        score = 0.0

        safe_count, frontier_count = self._reachable_safe_stats(grid, npos, future_blocked, future_danger, depth=8)
        safety_mult = 2.0 if self._safe_mode else 1.0
        score += 14.0 * safety_mult * safe_count + 7.0 * frontier_count

        min_danger = self._min_danger_time(npos, future_danger, horizon=9)
        score += 12.0 * safety_mult * min(min_danger, 10)
        if min_danger <= 2:
            score -= 320.0 * safety_mult
        elif min_danger <= 4:
            score -= 140.0 * safety_mult

        h, w = grid.shape
        cx, cy = (h - 1) / 2.0, (w - 1) / 2.0
        score -= 1.1 * (abs(npos[0] - cx) + abs(npos[1] - cy))

        cap = int(players[self.agent_id][3])
        bonus = int(players[self.agent_id][4])
        item_mult = 1.8 if self._push_items else 1.0
        item_score = self._nearest_item_score(grid, npos, future_blocked, future_danger, cap, bonus)
        score += item_mult * item_score
        if grid[npos[0], npos[1]] == self.ITEM_CAPACITY:
            score += 460.0
        elif grid[npos[0], npos[1]] == self.ITEM_RADIUS:
            score += 360.0

        farm_mult = 1.6 if self._push_boxes else 1.0
        if self._safe_mode:
            farm_mult = 0.4
        farm_score = self._nearest_bomb_spot_score(grid, npos, future_blocked, future_danger, my_radius)
        score += farm_mult * farm_score
        if self.turn < 180:
            score += 0.55 * farm_score

        if enemies:
            if self._safe_mode:
                # Leader lock-in: distance from enemies is worth points.
                d_enemy_m = min(abs(ep[0] - npos[0]) + abs(ep[1] - npos[1]) for ep in enemies)
                score += 14.0 * min(d_enemy_m, 7)
            else:
                d_enemy = self._nearest_distance(grid, npos, set(enemies), future_blocked, future_danger, max_depth=12)
                if d_enemy is not None:
                    power = int(players[self.agent_id][4]) + int(players[self.agent_id][3])
                    chase_weight = 5.0 + min(18.0, 3.0 * power) + (4.0 if self.turn > 180 else 0.0)
                    if self._push_kills:
                        chase_weight *= 2.2
                    score += max(0.0, 14.0 - d_enemy) * chase_weight

                # Behavioral vulnerability: approach clumsy enemies (only those).
                hunt_positions = {ep for eid, ep in zip(enemy_ids, enemies) if self._huntable(eid)}
                if hunt_positions:
                    d_hunt = self._nearest_distance(grid, npos, hunt_positions, future_blocked, future_danger, max_depth=14)
                    if d_hunt is not None:
                        expo = max(self._vuln(eid) for eid in enemy_ids if self._huntable(eid))
                        score += max(0.0, 14.0 - d_hunt) * 10.0 * min(expo, 4.0)

                if self._hunt_target is not None:
                    tp = None
                    for eid, ep in zip(enemy_ids, enemies):
                        if eid == self._hunt_target:
                            tp = ep
                    if tp is not None:
                        md = abs(tp[0] - npos[0]) + abs(tp[1] - npos[1])
                        score += max(0.0, 13.0 - md) * 22.0

                # Trap-setup approach bonus (kills are the top tie-breaker).
                if bombs_left > 0 and action != 5:
                    score += self._trap_setup_value(grid, players, bombs, npos, my_radius, enemy_ids, enemies, blocked)

            for eid, ep in zip(enemy_ids, enemies):
                md = abs(ep[0] - npos[0]) + abs(ep[1] - npos[1])
                enemy_power = int(players[eid][3]) + int(players[eid][4])
                my_power = int(players[self.agent_id][3]) + int(players[self.agent_id][4])
                if md <= 2 and enemy_power >= my_power and safe_count < 9:
                    score -= 120.0 / max(1, md)

        if action == 0:
            score -= 6.0
        else:
            score += 4.0

        if self._enemy_bomb_threat(grid, npos, players, enemy_ids):
            threat_limit = 15
            threat_penalty = 150.0 * safety_mult
            if safe_count < threat_limit or action == 0:
                score -= threat_penalty

        if self._joint_trap is not None:
            jdanger, jblocked = self._joint_trap
            if not self._survives_from(grid, npos, jblocked | set(bomb_cells_now), jdanger,
                                       horizon=8, allow_start_bomb=hypothetical_bomb):
                score -= 1600.0

        risk = self._enemy_hyp_bomb_trap_risk(grid, npos, players, bombs, enemy_ids, future_blocked)
        if risk >= 2:
            # An adjacent armed enemy holds a guaranteed kill on this cell.
            # Hard veto; tiny tie-breakers keep doomed alternatives ordered.
            return -10**10 + 10.0 * safe_count + min_danger
        elif risk == 1:
            # One enemy step away from sealing us in: outweigh any farm pull.
            score -= 2400.0
            if safe_count < 22:
                score -= 8.0 * (22 - safe_count)

        return score

    def _bomb_value(self, grid, pos, radius, enemies, players, bombs, danger, enemy_ids, eff_boxes=None):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        if eff_boxes is not None:
            boxes = eff_boxes
        else:
            boxes = sum(1 for t in blast if grid[t[0], t[1]] == self.BOX)
        value = 0.0
        value += 22.0
        value += 380.0 * boxes
        if boxes >= 2:
            value += 260.0
        if boxes == 0 and not any(ep in blast for ep in enemies):
            value -= 260.0

        for eid, ep in zip(enemy_ids, enemies):
            md = abs(ep[0] - pos[0]) + abs(ep[1] - pos[1])
            if ep in blast:
                value += 520.0
                # Saturate demonstrably clumsy enemies with blasts.
                if self._huntable(eid):
                    value += 140.0 * min(self._vuln(eid), 4.0)
                    # Graded trap pressure: clumsy enemy with at most one exit
                    # outside the blast rarely threads the needle.
                    bomb_cells = {(int(b[0]), int(b[1])) for b in bombs}
                    exits = sum(
                        1 for a in self.MOVE_ACTIONS
                        if self._passable(grid, *self._next(ep, a))
                        and self._next(ep, a) not in bomb_cells
                        and self._next(ep, a) not in blast
                    )
                    if exits <= 1:
                        value += 350.0
                if self._push_kills:
                    value += 220.0
                enemy_blocked = {(int(b[0]), int(b[1])) for b in bombs}
                enemy_blocked.discard(ep)
                can_enemy_escape = self._survives_from(grid, ep, enemy_blocked, danger, horizon=7)
                if not can_enemy_escape:
                    value += 900.0
            else:
                if md <= radius + 1:
                    value += max(0.0, 160.0 - 25.0 * md)
            # Blast saturation in a huntable enemy's region: even non-hitting
            # bombs block lanes and catch blunders from clumsy agents.
            if self._huntable(eid) and md <= 4:
                value += 70.0 * (5 - md)

        degree = self._free_degree(grid, pos, {(int(b[0]), int(b[1])) for b in bombs})
        if degree <= 1:
            value -= 80.0
        return value

    # ==================================================================
    # Danger and bomb model
    # ==================================================================

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
        out = []
        for b in bombs:
            out.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        out.append((int(pos[0]), int(pos[1]), int(timer), int(owner)))
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

    def _blast_tiles_nocache(self, grid, bx, by, radius):
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

    def _blast_tiles(self, grid, bx, by, radius):
        key = (int(bx), int(by), int(radius))
        cache = getattr(self, "_blast_cache", None)
        if cache is not None and key in cache:
            return cache[key]
        tiles = self._blast_tiles_nocache(grid, bx, by, radius)
        if cache is not None:
            cache[key] = tiles
        return tiles

    # ==================================================================
    # Search / pathfinding helpers
    # ==================================================================

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
                        continue
                else:
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

    def _distance_map(self, grid, start, blocked, danger=None, max_depth=99):
        q = deque([(start, 0)])
        dist = {start: 0}
        while q:
            pos, d = q.popleft()
            if d >= max_depth:
                continue
            for a in self.MOVE_ACTIONS:
                npos = self._next(pos, a)
                nd = d + 1
                if npos in dist:
                    continue
                if not self._passable(grid, *npos) or npos in blocked:
                    continue
                if danger is not None and npos in danger.get(nd, set()):
                    continue
                dist[npos] = nd
                q.append((npos, nd))
        return dist

    # ==================================================================
    # Tactical evaluation helpers
    # ==================================================================

    def _best_spot_within(self, grid, start, blocked, danger, radius, depth=2):
        """Max boxes hit from any cell within `depth` steps (defer-greed check)."""
        best = 0
        dist = self._distance_map(grid, start, blocked, danger, max_depth=depth)
        for pos, d in dist.items():
            if d == 0:
                continue
            best = max(best, self._count_boxes_in_blast(grid, pos, radius))
        return best

    def _nearest_item_score(self, grid, pos, blocked, danger, bombs_left, bomb_bonus):
        dist = self._distance_map(grid, pos, blocked, danger, max_depth=14)
        best = 0.0
        h, w = grid.shape
        for x in range(h):
            for y in range(w):
                cell = int(grid[x, y])
                if cell != self.ITEM_CAPACITY and cell != self.ITEM_RADIUS:
                    continue
                d = dist.get((x, y))
                if d is None:
                    continue
                base = 520.0 if cell == self.ITEM_CAPACITY else 430.0
                if cell == self.ITEM_CAPACITY and bombs_left <= 1:
                    base += 220.0
                if cell == self.ITEM_RADIUS and bomb_bonus <= 1:
                    base += 150.0
                best = max(best, base / (d + 1))
        return best

    def _nearest_bomb_spot_score(self, grid, pos, blocked, danger, radius):
        dist = self._distance_map(grid, pos, blocked, danger, max_depth=14)
        best = 0.0
        h, w = grid.shape
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if not self._passable(grid, x, y) or (x, y) in blocked:
                    continue
                d = dist.get((x, y))
                if d is None:
                    continue
                boxes = self._count_boxes_in_blast(grid, (x, y), radius)
                if boxes <= 0:
                    continue
                val = (360.0 * boxes + (160.0 if boxes >= 2 else 0.0)) / (d + 1)
                if d == 0:
                    val += 110.0 * boxes
                best = max(best, val)
        return best

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for x, y in self._blast_tiles(grid, pos[0], pos[1], radius) if int(grid[x, y]) == self.BOX)

    def _enemy_bomb_threat(self, grid, pos, players, enemy_ids):
        for eid in enemy_ids:
            ep = (int(players[eid][0]), int(players[eid][1]))
            if int(players[eid][3]) <= 0:
                continue
            er = max(1, min(5, int(players[eid][4]) + 1))
            if pos in self._blast_tiles(grid, ep[0], ep[1], er):
                return True
        return False

    def _enemy_hyp_bomb_trap_risk(self, grid, pos, players, bombs, enemy_ids, blocked):
        """Two-level pessimistic trap model.

        Returns 2 if an enemy bomb placed NOW (k=0) leaves us with no
        time-expanded escape from `pos` (the enemy holds a free-kill option),
        1 if an enemy bomb placed after ONE enemy step (k=1) would do so,
        0 otherwise. Hypothetical danger schedules are cached per turn.
        """
        worst = 0
        bomb_cells_now = {(int(b[0]), int(b[1])) for b in bombs}
        for eid in enemy_ids:
            if int(players[eid][2]) != 1 or int(players[eid][3]) <= 0:
                continue
            ep = (int(players[eid][0]), int(players[eid][1]))
            er = max(1, min(5, int(players[eid][4]) + 1))
            md = abs(ep[0] - pos[0]) + abs(ep[1] - pos[1])
            if md > er + 2:
                continue
            spots = [(ep, 0)]
            for a in self.MOVE_ACTIONS:
                q = self._next(ep, a)
                if self._passable(grid, *q) and q not in bomb_cells_now:
                    spots.append((q, 1))
            for spot, k in spots:
                if k == 1 and worst >= 1:
                    continue
                if spot in bomb_cells_now:
                    continue
                if pos not in self._blast_tiles(grid, spot[0], spot[1], er):
                    continue
                ck = (spot, er)
                cached = self._hyp_trap_cache.get(ck)
                if cached is None:
                    hyp_bombs = self._with_hypothetical_bomb(bombs, spot, eid, timer=7, radius=er)
                    hyp_danger, _ = self._danger_schedule(grid, hyp_bombs, players, horizon=8)
                    shifted = self._shift_danger(hyp_danger, 1)
                    self.bomb_radius_memory.pop((int(spot[0]), int(spot[1]), int(eid)), None)
                    # A rational enemy only places a seal bomb it can itself
                    # escape; suicidal threats are ignored.
                    enemy_ok = self._survives_from(
                        grid, spot, set(blocked) | {spot}, shifted, horizon=7, allow_start_bomb=True
                    )
                    cached = (shifted, enemy_ok)
                    self._hyp_trap_cache[ck] = cached
                shifted, enemy_ok = cached
                if not enemy_ok and not self._huntable(eid):
                    continue  # competent enemies do not place suicidal seals
                hyp_blocked = set(blocked) | {spot}
                if not self._survives_from(grid, pos, hyp_blocked, shifted, horizon=7):
                    if k == 0:
                        return 2
                    worst = 1
        return worst

    def _shift_danger(self, danger, offset):
        if offset <= 0:
            return danger
        max_t = max(danger.keys()) if danger else 0
        shifted = {}
        for t in range(1, max(1, max_t - offset) + 1):
            shifted[t] = set(danger.get(t + offset, set()))
        return shifted

    # ==================================================================
    # Primitive helpers
    # ==================================================================

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
