"""
Hybrid safety/search Bomberland agent for GDGoC-HCMUS AI Challenge 2026.

Design goals:
- Submission-safe: one file, no network, no file IO, CPU only.
- Strong baseline: time-indexed bomb danger, escape BFS, safe bombing, item/box farming,
  and small beam search over our own future actions.

Important engine quirk from the participant kit:
  action 1 -> row - 1, action 2 -> row + 1, action 3 -> col - 1, action 4 -> col + 1.
The public labels LEFT/RIGHT/UP/DOWN are misleading in engine/game.py.
"""

from collections import deque
import random
import time


class Agent:
    team_id = "SafeBeamAgent"

    # Engine action deltas, not semantic labels.
    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    MOVE_ACTIONS = (1, 2, 3, 4)
    ALL_STEP_ACTIONS = (0, 1, 2, 3, 4)
    PLACE_BOMB = 5

    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4

    MAX_RADIUS = 5
    DANGER_HORIZON = 13
    ESCAPE_HORIZON = 12
    BEAM_DEPTH = 3
    BEAM_WIDTH = 10
    TIME_BUDGET_SEC = 0.025

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step = 0
        self.last_safe_action = 0
        # Deterministic-ish tie-breaking per player, but not identical for all seats.
        self.rng = random.Random(1009 + self.agent_id * 9176)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def act(self, obs: dict) -> int:
        start_time = time.perf_counter()
        self.step += 1

        try:
            state = self._parse_obs(obs)
            if not state["alive"]:
                return 0

            records = self._current_bomb_records(state)
            danger, active = self._build_danger_and_active(state["grid"], records)
            my_pos = state["my_pos"]

            legal = self._legal_actions(state, active, t=1)
            if not legal:
                return 0

            safe = self._safe_actions(state, records, legal)
            if not safe:
                return self._least_bad_action(state, records, legal)

            # If any bomb will hit our current tile soon, escape dominates everything.
            if self._threat_time(my_pos, danger, max_t=6) is not None:
                action = self._best_escape_action(state, records, safe)
                self.last_safe_action = action
                return int(action)

            # Opportunistic but safe immediate bomb: enemy trap > box farming.
            bomb_action = self._try_immediate_bomb(state, records, safe)
            if bomb_action is not None:
                self.last_safe_action = bomb_action
                return int(bomb_action)

            # Small own-action beam search. If it gets too close to time budget,
            # it returns the best partial first action seen so far.
            action = self._beam_search_action(state, records, safe, start_time)
            if action in safe:
                self.last_safe_action = action
                return int(action)

            # Targeted deterministic fallback.
            action = self._target_fallback_action(state, records, safe)
            self.last_safe_action = action
            return int(action)
        except Exception:
            # In competition, failing to act is worse than a conservative stop.
            return int(self.last_safe_action) if self.last_safe_action in (0, 1, 2, 3, 4, 5) else 0

    # ---------------------------------------------------------------------
    # Observation parsing and primitive helpers
    # ---------------------------------------------------------------------

    def _parse_obs(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        alive = self.agent_id < len(players) and int(players[self.agent_id][2]) == 1
        if alive:
            p = players[self.agent_id]
            my_pos = (int(p[0]), int(p[1]))
            bombs_left = int(p[3])
            radius = max(1, min(self.MAX_RADIUS, 1 + int(p[4])))
        else:
            my_pos = (1, 1)
            bombs_left = 0
            radius = 1

        enemies = []
        for i, p in enumerate(players):
            if i == self.agent_id or int(p[2]) != 1:
                continue
            enemies.append({
                "id": int(i),
                "pos": (int(p[0]), int(p[1])),
                "bombs_left": int(p[3]),
                "radius": max(1, min(self.MAX_RADIUS, 1 + int(p[4]))),
            })

        bomb_positions = set()
        for b in bombs:
            bomb_positions.add((int(b[0]), int(b[1])))

        return {
            "grid": grid,
            "players": players,
            "bombs": bombs,
            "alive": alive,
            "my_pos": my_pos,
            "bombs_left": bombs_left,
            "radius": radius,
            "enemies": enemies,
            "bomb_positions": bomb_positions,
            "height": int(grid.shape[0]),
            "width": int(grid.shape[1]),
        }

    def _current_bomb_records(self, state):
        """Bomb records use absolute future time from now.

        natural_t=1 means the bomb explodes after our current action is resolved.
        placed_t=0 means the bomb already exists before our current movement.
        """
        records = []
        players = state["players"]
        for b in state["bombs"]:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner = int(b[3]) if len(b) > 3 else -1
            radius = 2
            if 0 <= owner < len(players):
                radius = max(1, min(self.MAX_RADIUS, 1 + int(players[owner][4])))
            records.append({
                "pos": (bx, by),
                "radius": radius,
                "owner": owner,
                "placed_t": 0,
                "natural_t": max(1, timer),
            })
        return records

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable_grid(self, grid, x, y):
        return self._in_bounds(grid, x, y) and int(grid[x, y]) in (self.GRASS, self.ITEM_RADIUS, self.ITEM_CAPACITY)

    def _next_pos(self, pos, action):
        if action == self.PLACE_BOMB:
            return pos
        dx, dy = self.MOVES.get(int(action), (0, 0))
        return (pos[0] + dx, pos[1] + dy)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, pos):
        for a in self.MOVE_ACTIONS:
            yield a, self._next_pos(pos, a)

    # ---------------------------------------------------------------------
    # Bomb / danger model
    # ---------------------------------------------------------------------

    def _blast_tiles(self, grid, pos, radius):
        bx, by = pos
        tiles = {(bx, by)}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for r in range(1, int(radius) + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == self.WALL:
                    break
                tiles.add((x, y))
                if cell == self.BOX:
                    break
        return tiles

    def _build_danger_and_active(self, grid, records, horizon=None):
        """Return (danger[t], active_bomb_positions[t]) for t in [0..horizon].

        Handles chain reactions conservatively. Future bombs have placed_t > 0 and
        cannot be triggered before they exist.
        """
        if horizon is None:
            horizon = self.DANGER_HORIZON

        n = len(records)
        danger = [set() for _ in range(horizon + 1)]
        active = [set() for _ in range(horizon + 1)]
        if n == 0:
            return danger, active

        blasts = [self._blast_tiles(grid, rec["pos"], rec["radius"]) for rec in records]
        explode_t = [max(int(rec["natural_t"]), int(rec.get("placed_t", 0)) + 1) for rec in records]
        placed_t = [int(rec.get("placed_t", 0)) for rec in records]

        changed = True
        passes = 0
        while changed and passes < n + 2:
            changed = False
            passes += 1
            for src in range(n):
                ts = explode_t[src]
                for dst in range(n):
                    if src == dst:
                        continue
                    # dst must exist by the time src explodes.
                    if ts < placed_t[dst]:
                        continue
                    if ts <= explode_t[dst] and records[dst]["pos"] in blasts[src]:
                        if explode_t[dst] != ts:
                            explode_t[dst] = ts
                            changed = True

        for i, rec in enumerate(records):
            et = explode_t[i]
            # A bomb blocks movement on steps after it has been placed and before
            # or on the step it explodes. Placement itself happens after movement.
            for t in range(max(1, placed_t[i] + 1), min(horizon, et) + 1):
                active[t].add(rec["pos"])
            if 1 <= et <= horizon:
                danger[et].update(blasts[i])

        return danger, active

    def _threat_time(self, pos, danger, max_t=None):
        if max_t is None:
            max_t = len(danger) - 1
        max_t = min(max_t, len(danger) - 1)
        for t in range(1, max_t + 1):
            if pos in danger[t]:
                return t
        return None

    def _danger_union(self, danger, lo=1, hi=None):
        if hi is None:
            hi = len(danger) - 1
        hi = min(hi, len(danger) - 1)
        out = set()
        for t in range(max(1, lo), hi + 1):
            out.update(danger[t])
        return out

    # ---------------------------------------------------------------------
    # Legal/safe action filtering
    # ---------------------------------------------------------------------

    def _walkable_at(self, grid, pos, active, t):
        x, y = pos
        if not self._passable_grid(grid, x, y):
            return False
        if 0 <= t < len(active) and pos in active[t]:
            return False
        return True

    def _legal_actions(self, state, active, t=1, pos=None, bombs_left=None):
        grid = state["grid"]
        if pos is None:
            pos = state["my_pos"]
        if bombs_left is None:
            bombs_left = state["bombs_left"]

        actions = [0]
        for a, npos in self._neighbors(pos):
            if self._walkable_at(grid, npos, active, t):
                actions.append(a)

        # Existing bombs block placement at this tile.
        blocked_by_bomb = 0 <= t < len(active) and pos in active[t]
        if bombs_left > 0 and not blocked_by_bomb:
            actions.append(self.PLACE_BOMB)
        return actions

    def _records_with_my_bomb(self, state, records, placed_t=1, pos=None):
        if pos is None:
            pos = state["my_pos"]
        new_records = list(records)
        new_records.append({
            "pos": pos,
            "radius": state["radius"],
            "owner": self.agent_id,
            "placed_t": int(placed_t),
            # New bombs are appended then ticked during the same step; they explode
            # 7 decision steps after placement in this absolute-time convention.
            "natural_t": int(placed_t) + 6,
        })
        return new_records

    def _safe_actions(self, state, records, legal):
        safe = []
        for action in legal:
            if action == self.PLACE_BOMB:
                test_records = self._records_with_my_bomb(state, records, placed_t=1, pos=state["my_pos"])
                danger, active = self._build_danger_and_active(state["grid"], test_records)
                npos = state["my_pos"]
            else:
                test_records = records
                danger, active = self._build_danger_and_active(state["grid"], test_records)
                npos = self._next_pos(state["my_pos"], action)

            if 1 < len(danger) and npos in danger[1]:
                continue

            surv = self._survival_info(state["grid"], npos, danger, active, start_t=1, horizon=self.ESCAPE_HORIZON)
            # Require either reaching horizon, or at least many steps when horizon is truncated
            # by dense bomb chaos. This avoids suicidal but superficially safe moves.
            if surv["max_t"] >= min(8, self.ESCAPE_HORIZON) or surv["horizon_cells"] > 0:
                safe.append(action)

        # Keep at least non-lethal one-step movements if the strict filter is too pessimistic.
        if not safe:
            base_danger, _ = self._build_danger_and_active(state["grid"], records)
            for action in legal:
                if action == self.PLACE_BOMB:
                    continue
                npos = self._next_pos(state["my_pos"], action)
                if len(base_danger) <= 1 or npos not in base_danger[1]:
                    safe.append(action)
        return safe

    def _least_bad_action(self, state, records, legal):
        danger, active = self._build_danger_and_active(state["grid"], records)
        best = 0
        best_score = -10**18
        for a in legal:
            if a == self.PLACE_BOMB:
                continue
            pos = self._next_pos(state["my_pos"], a)
            score = 0
            tt = self._threat_time(pos, danger, max_t=self.DANGER_HORIZON)
            score += 1000 * (self.DANGER_HORIZON + 1 if tt is None else tt)
            score += 40 * self._open_neighbor_count(state["grid"], pos, active, t=1)
            score += self._reachable_count(state["grid"], pos, active, danger, start_t=1, depth=5)
            if pos in danger[1]:
                score -= 100000
            if score > best_score:
                best_score = score
                best = a
        return best

    # ---------------------------------------------------------------------
    # Time-aware survival BFS
    # ---------------------------------------------------------------------

    def _survival_info(self, grid, start, danger, active, start_t=1, horizon=12):
        horizon = min(horizon, len(danger) - 1, len(active) - 1)
        if start_t > horizon:
            return {"max_t": start_t, "horizon_cells": 1, "seen_count": 1}
        if start in danger[start_t]:
            return {"max_t": start_t - 1, "horizon_cells": 0, "seen_count": 0}
        if not self._passable_grid(grid, start[0], start[1]):
            return {"max_t": start_t - 1, "horizon_cells": 0, "seen_count": 0}

        q = deque([(start, start_t)])
        seen = {(start, start_t)}
        max_t = start_t
        horizon_positions = set()

        while q:
            pos, t = q.popleft()
            max_t = max(max_t, t)
            if t >= horizon:
                horizon_positions.add(pos)
                continue

            nt = t + 1
            for a in self.ALL_STEP_ACTIONS:
                npos = self._next_pos(pos, a)
                if not self._walkable_at(grid, npos, active, nt):
                    continue
                if npos in danger[nt]:
                    continue
                key = (npos, nt)
                if key in seen:
                    continue
                seen.add(key)
                q.append((npos, nt))

        return {"max_t": max_t, "horizon_cells": len(horizon_positions), "seen_count": len(seen)}

    def _first_actions_to_survive(self, grid, start, danger, active, start_t=1, horizon=12):
        """Return candidate first actions and scores for escape mode."""
        horizon = min(horizon, len(danger) - 1, len(active) - 1)
        out = []
        for action in self.ALL_STEP_ACTIONS:
            npos = self._next_pos(start, action)
            nt = start_t
            if not self._walkable_at(grid, npos, active, nt):
                continue
            if npos in danger[nt]:
                continue
            info = self._survival_info(grid, npos, danger, active, start_t=nt, horizon=horizon)
            score = 10000 * info["horizon_cells"] + 1000 * info["max_t"] + info["seen_count"]
            score += 15 * self._open_neighbor_count(grid, npos, active, nt)
            if action == 0:
                score -= 120
            out.append((score, action))
        out.sort(reverse=True)
        return out

    # ---------------------------------------------------------------------
    # Tactical action choices
    # ---------------------------------------------------------------------

    def _best_escape_action(self, state, records, safe):
        danger, active = self._build_danger_and_active(state["grid"], records)
        candidates = self._first_actions_to_survive(
            state["grid"], state["my_pos"], danger, active, start_t=1, horizon=self.ESCAPE_HORIZON
        )
        for _, action in candidates:
            if action in safe and action != self.PLACE_BOMB:
                return action
        non_bomb = [a for a in safe if a != self.PLACE_BOMB]
        return non_bomb[0] if non_bomb else safe[0]

    def _try_immediate_bomb(self, state, records, safe):
        if self.PLACE_BOMB not in safe:
            return None
        value = self._bomb_value(state, records, state["my_pos"], placed_t=1)
        # Early-game farming can be aggressive; late game requires more tactical value.
        threshold = 1800 if self.step < 130 else 3200
        if value >= threshold:
            return self.PLACE_BOMB
        return None

    def _bomb_value(self, state, records, pos, placed_t=1):
        grid = state["grid"]
        radius = state["radius"]
        blast = self._blast_tiles(grid, pos, radius)
        boxes = sum(1 for p in blast if int(grid[p[0], p[1]]) == self.BOX)

        value = 0
        value += 2100 * boxes

        enemies = state["enemies"]
        for e in enemies:
            epos = e["pos"]
            dist = self._manhattan(pos, epos)
            if epos in blast:
                # Direct hit is only good if the enemy has constrained exits.
                escape_cells = self._enemy_escape_cells_after_bomb(state, records, epos, pos, radius, placed_t)
                value += 9000
                value += max(0, 7 - escape_cells) * 2600
            elif dist <= radius + 2:
                # Pressure by reducing nearby safe cells.
                escape_cells = self._enemy_escape_cells_after_bomb(state, records, epos, pos, radius, placed_t)
                value += max(0, 5 - escape_cells) * 1100

        # Prefer bombs that create future item/space, but penalize low self mobility.
        test_records = self._records_with_my_bomb(state, records, placed_t=placed_t, pos=pos)
        danger, active = self._build_danger_and_active(grid, test_records)
        info = self._survival_info(grid, pos, danger, active, start_t=placed_t, horizon=self.ESCAPE_HORIZON)
        if info["horizon_cells"] <= 0 and info["max_t"] < 8:
            value -= 100000
        else:
            value += 200 * info["horizon_cells"] + 20 * info["seen_count"]

        # Bombing with no box/enemy pressure is usually a waste and can self-trap.
        if boxes == 0 and all(e["pos"] not in blast for e in enemies):
            value -= 3000
        return value

    def _enemy_escape_cells_after_bomb(self, state, records, enemy_pos, bomb_pos, radius, placed_t):
        # Conservative small estimate: add our hypothetical bomb and count positions
        # the enemy could occupy a few steps later. Enemy can overlap players, but not bombs/walls/boxes.
        recs = list(records)
        recs.append({
            "pos": bomb_pos,
            "radius": radius,
            "owner": self.agent_id,
            "placed_t": placed_t,
            "natural_t": placed_t + 6,
        })
        danger, active = self._build_danger_and_active(state["grid"], recs)
        info = self._survival_info(state["grid"], enemy_pos, danger, active, start_t=1, horizon=7)
        return int(info["horizon_cells"] + info["seen_count"] // 12)

    # ---------------------------------------------------------------------
    # Beam search over own actions
    # ---------------------------------------------------------------------

    def _beam_search_action(self, state, records, initial_safe, start_time):
        grid = state["grid"]
        root_pos = state["my_pos"]
        root_score = self._eval_position(state, records, root_pos, t=0, collected_bonus=0, boxes_bonus=0)
        best_first = initial_safe[0]
        best_score = -10**18

        # Node tuple: (score, pos, t, bombs_left, extra_records_tuple, first_action, collected, boxes)
        # extra_records_tuple stores dict-like tuples: (pos, radius, owner, placed_t, natural_t)
        beam = [(root_score, root_pos, 0, state["bombs_left"], tuple(), None, 0, 0)]
        seen_by_layer = set()

        for depth in range(self.BEAM_DEPTH):
            if time.perf_counter() - start_time > self.TIME_BUDGET_SEC:
                break

            candidates = []
            for score, pos, t, bombs_left, extra_tuple, first_action, collected, boxes_bonus in beam:
                extra_records = [
                    {"pos": er[0], "radius": er[1], "owner": er[2], "placed_t": er[3], "natural_t": er[4]}
                    for er in extra_tuple
                ]
                all_records = list(records) + extra_records
                danger, active = self._build_danger_and_active(grid, all_records)
                nt = t + 1
                if nt >= len(danger):
                    continue

                # Regain a bomb when our simulated bombs have already exploded.
                effective_bombs_left = bombs_left
                for er in extra_tuple:
                    if er[4] <= t and er[2] == self.agent_id:
                        effective_bombs_left = max(effective_bombs_left, 1)

                legal = self._legal_actions(state, active, t=nt, pos=pos, bombs_left=effective_bombs_left)

                for action in legal:
                    if first_action is None and action not in initial_safe:
                        continue

                    npos = pos if action == self.PLACE_BOMB else self._next_pos(pos, action)
                    new_extra_tuple = extra_tuple
                    new_bombs_left = effective_bombs_left
                    add_score = 0
                    new_collected = collected
                    new_boxes_bonus = boxes_bonus

                    if action == self.PLACE_BOMB:
                        # Avoid low-value simulated bombs; they bloat search and cause traps.
                        temp_state = dict(state)
                        temp_state["my_pos"] = pos
                        bv = self._bomb_value(temp_state, all_records, pos, placed_t=nt)
                        if bv < 1000:
                            continue
                        new_extra_tuple = extra_tuple + ((pos, state["radius"], self.agent_id, nt, nt + 6),)
                        new_bombs_left = max(0, effective_bombs_left - 1)
                        add_score += bv
                        blast = self._blast_tiles(grid, pos, state["radius"])
                        new_boxes_bonus += sum(1 for p in blast if int(grid[p[0], p[1]]) == self.BOX)
                    else:
                        if npos in danger[nt]:
                            continue
                        if action != 0 and not self._walkable_at(grid, npos, active, nt):
                            continue
                        cell = int(grid[npos[0], npos[1]])
                        if cell in (self.ITEM_RADIUS, self.ITEM_CAPACITY):
                            new_collected += 1
                            add_score += 5200 if cell == self.ITEM_CAPACITY and state["bombs_left"] <= 1 else 4300

                    # Rebuild after adding new bomb, then test survival from the new state.
                    final_records = list(records) + [
                        {"pos": er[0], "radius": er[1], "owner": er[2], "placed_t": er[3], "natural_t": er[4]}
                        for er in new_extra_tuple
                    ]
                    ndanger, nactive = self._build_danger_and_active(grid, final_records)
                    if nt < len(ndanger) and npos in ndanger[nt]:
                        continue
                    surv = self._survival_info(grid, npos, ndanger, nactive, start_t=nt, horizon=self.ESCAPE_HORIZON)
                    if surv["max_t"] < min(self.ESCAPE_HORIZON, nt + 5) and surv["horizon_cells"] == 0:
                        continue

                    first = action if first_action is None else first_action
                    pos_score = self._eval_position(state, final_records, npos, t=nt,
                                                    collected_bonus=new_collected,
                                                    boxes_bonus=new_boxes_bonus)
                    total = score * 0.15 + pos_score + add_score
                    total += 650 * surv["horizon_cells"] + 18 * surv["seen_count"] + 180 * surv["max_t"]
                    # Slightly prefer decisive earlier actions.
                    total -= depth * 8
                    if action == 0:
                        total -= 60

                    key = (nt, npos, new_bombs_left, first, len(new_extra_tuple))
                    if key in seen_by_layer:
                        continue
                    seen_by_layer.add(key)
                    candidates.append((total, npos, nt, new_bombs_left, new_extra_tuple, first, new_collected, new_boxes_bonus))

                    if total > best_score and first is not None:
                        best_score = total
                        best_first = first

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:self.BEAM_WIDTH]

        return best_first if best_first in initial_safe else initial_safe[0]

    # ---------------------------------------------------------------------
    # Evaluation and target helpers
    # ---------------------------------------------------------------------

    def _eval_position(self, state, records, pos, t=0, collected_bonus=0, boxes_bonus=0):
        grid = state["grid"]
        danger, active = self._build_danger_and_active(grid, records)
        score = 0

        tt = self._threat_time(pos, danger, max_t=self.DANGER_HORIZON)
        if tt is None:
            score += 9000
        else:
            score += 600 * tt
            if tt <= 2:
                score -= 12000

        reachable = self._reachable_count(grid, pos, active, danger, start_t=max(1, t), depth=6)
        score += 120 * reachable
        score += 250 * self._open_neighbor_count(grid, pos, active, min(max(1, t), len(active) - 1))

        cell = int(grid[pos[0], pos[1]])
        if cell == self.ITEM_CAPACITY:
            score += 3800 if state["bombs_left"] <= 1 else 2600
        elif cell == self.ITEM_RADIUS:
            score += 3000 if state["radius"] <= 2 else 2100

        # Distance to safe item.
        items = self._item_tiles(grid)
        if items:
            d_item = self._bfs_distance(grid, pos, items, active, danger, start_t=max(1, t), max_depth=10)
            if d_item is not None:
                score += max(0, 10 - d_item) * 360

        # Distance to high-value bombing spot.
        bomb_spots = self._box_bomb_spots(state, active, danger)
        if bomb_spots:
            d_box = self._bfs_distance(grid, pos, bomb_spots, active, danger, start_t=max(1, t), max_depth=10)
            if d_box is not None:
                score += max(0, 10 - d_box) * 210

        # Enemy pressure: approach when safe and powered; avoid hugging enemies in dead ends.
        enemies = state["enemies"]
        if enemies:
            nearest = min(self._manhattan(pos, e["pos"]) for e in enemies)
            if state["radius"] >= 2 or state["bombs_left"] >= 2:
                score += max(0, 8 - nearest) * 120
            if nearest <= 1 and reachable < 8:
                score -= 1800

        score += 1800 * collected_bonus + 450 * boxes_bonus
        # Avoid boundary/corner camping when not forced.
        if pos[0] in (1, state["height"] - 2) and pos[1] in (1, state["width"] - 2):
            score -= 250
        return score

    def _target_fallback_action(self, state, records, safe):
        grid = state["grid"]
        danger, active = self._build_danger_and_active(grid, records)
        my_pos = state["my_pos"]

        # Prefer items when safe.
        item_tiles = self._item_tiles(grid)
        action = self._move_to_targets(grid, my_pos, item_tiles, active, danger, allowed_first=safe)
        if action is not None:
            return action

        # Then move to a safe bombing spot.
        box_spots = self._box_bomb_spots(state, active, danger)
        action = self._move_to_targets(grid, my_pos, box_spots, active, danger, allowed_first=safe)
        if action is not None:
            return action

        # Then pressure nearest enemy without stepping into danger.
        enemy_tiles = {e["pos"] for e in state["enemies"]}
        action = self._move_to_targets(grid, my_pos, enemy_tiles, active, danger, allowed_first=safe, allow_target_occupied=True)
        if action is not None:
            return action

        # Finally pick the safest open move.
        best = safe[0]
        best_score = -10**18
        for action in safe:
            if action == self.PLACE_BOMB:
                continue
            npos = self._next_pos(my_pos, action)
            score = self._eval_position(state, records, npos, t=1)
            if score > best_score:
                best_score = score
                best = action
        return best

    def _item_tiles(self, grid):
        out = set()
        h, w = int(grid.shape[0]), int(grid.shape[1])
        for x in range(h):
            for y in range(w):
                if int(grid[x, y]) in (self.ITEM_RADIUS, self.ITEM_CAPACITY):
                    out.add((x, y))
        return out

    def _box_bomb_spots(self, state, active, danger):
        grid = state["grid"]
        spots = set()
        h, w = int(grid.shape[0]), int(grid.shape[1])
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if int(grid[x, y]) not in (self.GRASS, self.ITEM_RADIUS, self.ITEM_CAPACITY):
                    continue
                pos = (x, y)
                if pos in active[1] or pos in danger[1]:
                    continue
                boxes = self._count_boxes_in_blast(grid, pos, state["radius"])
                if boxes > 0:
                    spots.add(pos)
        return spots

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for p in self._blast_tiles(grid, pos, radius) if int(grid[p[0], p[1]]) == self.BOX)

    def _line_clear(self, grid, a, b):
        ax, ay = a
        bx, by = b
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if int(grid[ax, y]) in (self.WALL, self.BOX):
                    return False
            return True
        if ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if int(grid[x, ay]) in (self.WALL, self.BOX):
                    return False
            return True
        return False

    def _open_neighbor_count(self, grid, pos, active, t):
        cnt = 0
        t = min(max(0, t), len(active) - 1)
        for _, npos in self._neighbors(pos):
            if self._walkable_at(grid, npos, active, t):
                cnt += 1
        return cnt

    def _reachable_count(self, grid, start, active, danger, start_t=1, depth=6):
        start_t = min(max(1, start_t), len(active) - 1, len(danger) - 1)
        q = deque([(start, start_t, 0)])
        seen = {(start, start_t)}
        positions = {start}
        while q:
            pos, t, d = q.popleft()
            if d >= depth:
                continue
            nt = min(t + 1, len(active) - 1, len(danger) - 1)
            for a in self.ALL_STEP_ACTIONS:
                npos = self._next_pos(pos, a)
                if not self._walkable_at(grid, npos, active, nt):
                    continue
                if npos in danger[nt]:
                    continue
                key = (npos, nt)
                if key in seen:
                    continue
                seen.add(key)
                positions.add(npos)
                q.append((npos, nt, d + 1))
        return len(positions)

    def _bfs_distance(self, grid, start, targets, active, danger, start_t=1, max_depth=10):
        if not targets:
            return None
        start_t = min(max(1, start_t), len(active) - 1, len(danger) - 1)
        q = deque([(start, start_t, 0)])
        seen = {(start, start_t)}
        while q:
            pos, t, d = q.popleft()
            if pos in targets:
                return d
            if d >= max_depth:
                continue
            nt = min(t + 1, len(active) - 1, len(danger) - 1)
            for a in self.ALL_STEP_ACTIONS:
                npos = self._next_pos(pos, a)
                if not self._walkable_at(grid, npos, active, nt):
                    continue
                if npos in danger[nt]:
                    continue
                key = (npos, nt)
                if key in seen:
                    continue
                seen.add(key)
                q.append((npos, nt, d + 1))
        return None

    def _move_to_targets(self, grid, start, targets, active, danger, allowed_first=None, max_depth=12, allow_target_occupied=False):
        if not targets:
            return None
        allowed_first = set(allowed_first) if allowed_first is not None else set(self.MOVE_ACTIONS)
        q = deque([(start, 0, None)])
        seen = {start}
        while q:
            pos, d, first = q.popleft()
            if d > 0 and pos in targets:
                return first if first in allowed_first else None
            if d >= max_depth:
                continue
            nt = min(d + 1, len(active) - 1, len(danger) - 1)
            for a in self.MOVE_ACTIONS:
                npos = self._next_pos(pos, a)
                if npos in seen:
                    continue
                if not self._walkable_at(grid, npos, active, nt):
                    # Enemy target tiles are still passable in the real engine, but
                    # walls/boxes/bombs are never passable.
                    if not (allow_target_occupied and npos in targets and self._passable_grid(grid, npos[0], npos[1])):
                        continue
                if npos in danger[nt]:
                    continue
                nf = a if first is None else first
                if first is None and nf not in allowed_first:
                    continue
                seen.add(npos)
                q.append((npos, d + 1, nf))
        return None
