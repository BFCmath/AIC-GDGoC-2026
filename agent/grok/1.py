import random
from collections import deque
import numpy as np
from typing import Tuple, Set, List, Optional, Dict

class SmartBomberAgent:
    """
    Advanced Hybrid Agent - Kết hợp Rule-based + Short-term Search + Opportunistic Strategy
    Phù hợp cho online ladder (TrueSkill), inference nhanh (<80ms/step trên CPU).
    """
    MOVES = {
        0: (0, 0),   # STOP
        1: (-1, 0),  # LEFT
        2: (1, 0),   # RIGHT
        3: (0, -1),  # UP
        4: (0, 1),   # DOWN
    }
    
    team_id = "SmartBomberAgent_v1"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.last_mode = "explore"
        self.consecutive_safes = 0

    def act(self, observation: dict) -> int:
        grid = observation["map"]          # numpy array
        players = observation["players"]   # list of [x, y, alive, bombs_left, radius_bonus]
        bombs = observation["bombs"]       # list of [x, y, timer, owner_id?]

        if self.agent_id >= len(players) or not players[self.agent_id][2]:
            return 0

        my_x, my_y, alive, bombs_left, radius_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        my_radius = max(1, int(radius_bonus) + 1)
        my_bombs_left = int(bombs_left)

        # Precompute
        danger_now, danger_soon, danger_future = self._compute_danger_maps(grid, bombs, players)
        blocked = self._get_blocked_positions(players, bombs, my_pos)
        enemies = self._get_enemies(players, my_pos)

        valid_moves = self._get_valid_moves(grid, my_pos, blocked)

        # === 1. REACTIVE SAFETY (ưu tiên tuyệt đối) ===
        if danger_now[my_pos]:
            action = self._escape_immediate(grid, my_pos, blocked, danger_now, danger_soon)
            if action is not None:
                return action

        if danger_soon[my_pos]:
            action = self._find_safe_path(grid, my_pos, blocked, danger_soon, max_depth=6)
            if action is not None:
                return action

        # === 2. SHORT-TERM PLANNING (Core strength) ===
        best_action = self._short_term_planning(
            grid, my_pos, my_radius, my_bombs_left,
            blocked, danger_soon, danger_future, enemies, players
        )
        if best_action is not None:
            return best_action

        # === 3. STRATEGIC BEHAVIORS ===
        # Pick item
        item_action = self._go_to_best_item(grid, my_pos, blocked, danger_soon, my_bombs_left, radius_bonus)
        if item_action is not None:
            return item_action

        # Bombing opportunity
        if my_bombs_left > 0 and my_pos not in self._get_bomb_positions(bombs):
            if self._should_place_bomb_here(grid, my_pos, my_radius, enemies, blocked, danger_now):
                return 5

        # Move to good bombing position
        bomb_spot_action = self._move_to_good_bomb_spot(grid, my_pos, blocked, danger_soon, my_radius)
        if bomb_spot_action is not None:
            return bomb_spot_action

        # Fallback: safe exploration
        return self._safe_explore(grid, my_pos, blocked, danger_soon, valid_moves)

    # ====================== HELPER METHODS ======================

    def _compute_danger_maps(self, grid, bombs, players):
        """Tính danger_now, danger_soon, danger_future (3-7 steps)"""
        h, w = grid.shape
        danger_now = np.zeros((h, w), dtype=bool)
        danger_soon = np.zeros((h, w), dtype=bool)
        danger_future = np.zeros((h, w), dtype=bool)

        for b in bombs:
            bx, by, timer, *rest = b
            bx, by, timer = int(bx), int(by), int(timer)
            if timer <= 0:
                continue
            owner = int(rest[0]) if rest else -1
            radius = max(1, int(players[owner][4]) + 1) if 0 <= owner < len(players) else 2

            blast = self._get_blast_tiles(grid, bx, by, radius)

            for x, y in blast:
                if timer <= 1:
                    danger_now[x, y] = True
                if timer <= 3:
                    danger_soon[x, y] = True
                if timer <= 7:
                    danger_future[x, y] = True

        return danger_now, danger_soon, danger_future

    def _get_blast_tiles(self, grid, bx: int, by: int, radius: int) -> Set[Tuple[int, int]]:
        tiles = {(bx, by)}
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                    break
                tiles.add((x, y))
                if grid[x, y] in [1, 2]:   # wall or box stops blast
                    break
        return tiles

    def _get_blocked_positions(self, players, bombs, my_pos):
        occupied = {(int(p[0]), int(p[1])) for i, p in enumerate(players) 
                   if i != self.agent_id and p[2] == 1}
        bomb_pos = {(int(b[0]), int(b[1])) for b in bombs}
        blocked = occupied | bomb_pos
        blocked.discard(my_pos)
        return blocked

    def _get_enemies(self, players, my_pos):
        return [(int(p[0]), int(p[1])) for i, p in enumerate(players) 
                if i != self.agent_id and p[2] == 1]

    def _get_bomb_positions(self, bombs):
        return {(int(b[0]), int(b[1])) for b in bombs}

    def _get_valid_moves(self, grid, pos, blocked):
        actions = [0]
        for a in [1,2,3,4]:
            nx, ny = pos[0] + self.MOVES[a][0], pos[1] + self.MOVES[a][1]
            if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 
                grid[nx, ny] not in [1, 2] and (nx, ny) not in blocked):
                actions.append(a)
        return actions

    # ====================== PLANNING ======================

    def _short_term_planning(self, grid, my_pos, my_radius, my_bombs_left,
                           blocked, danger_soon, danger_future, enemies, players):
        """Simple Rolling Horizon / Greedy Search 5-7 steps"""
        best_score = -9999
        best_action = None

        for first_action in [0,1,2,3,4]:
            if first_action not in self._get_valid_moves(grid, my_pos, blocked):
                continue

            score = self._simulate_sequence(grid, my_pos, first_action, my_radius, 
                                          my_bombs_left, blocked, danger_soon, enemies, depth=6)
            if score > best_score:
                best_score = score
                best_action = first_action

        return best_action if best_action is not None else None

    def _simulate_sequence(self, grid, pos, first_action, my_radius, my_bombs_left,
                          blocked, danger_soon, enemies, depth=6) -> float:
        """Đánh giá sequence đơn giản"""
        score = 0
        current_pos = (pos[0] + self.MOVES[first_action][0], 
                      pos[1] + self.MOVES[first_action][1])

        # Survival bonus
        if not danger_soon[current_pos]:
            score += 40

        # Item nearby
        if self._is_item_near(grid, current_pos, 3):
            score += 25

        # Kill potential
        for ex, ey in enemies:
            dist = abs(ex - current_pos[0]) + abs(ey - current_pos[1])
            if dist <= my_radius + 2:
                score += 30 if dist <= my_radius else 10

        # Box breaking potential
        boxes = self._count_boxes_in_blast(grid, current_pos, my_radius)
        score += boxes * 12

        return score

    # ====================== TACTICAL HELPERS ======================

    def _escape_immediate(self, grid, pos, blocked, danger_now, danger_soon):
        """Tìm đường thoát khẩn cấp"""
        for a in [1,2,3,4]:
            npos = (pos[0] + self.MOVES[a][0], pos[1] + self.MOVES[a][1])
            if (self._is_passable(grid, npos) and npos not in blocked and 
                not danger_now[npos]):
                return a
        return None

    def _find_safe_path(self, grid, start, blocked, danger_soon, max_depth=8):
        """BFS tìm safe tile"""
        q = deque([(start, None)])
        seen = {start}
        while q:
            pos, first_action = q.popleft()
            if first_action is not None and not danger_soon[pos]:
                return first_action

            if len(seen) > 30:  # limit computation
                break

            for a in [1,2,3,4]:
                npos = (pos[0] + self.MOVES[a][0], pos[1] + self.MOVES[a][1])
                if npos in seen or not self._is_passable(grid, npos) or npos in blocked:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))
        return None

    def _should_place_bomb_here(self, grid, pos, radius, enemies, blocked, danger_now) -> bool:
        boxes = self._count_boxes_in_blast(grid, pos, radius)
        hits_enemy = any(abs(ex-pos[0]) + abs(ey-pos[1]) <= radius for ex,ey in enemies)
        
        # Escape check
        can_escape = False
        for a in [1,2,3,4]:
            npos = (pos[0] + self.MOVES[a][0], pos[1] + self.MOVES[a][1])
            if self._is_passable(grid, npos) and npos not in blocked and not danger_now[npos]:
                can_escape = True
                break

        return (boxes >= 1 or hits_enemy) and can_escape

    def _count_boxes_in_blast(self, grid, pos, radius):
        return len([t for t in self._get_blast_tiles(grid, pos[0], pos[1], radius) 
                   if grid[t[0], t[1]] == 2])

    def _go_to_best_item(self, grid, pos, blocked, danger_soon, bombs_left, radius_bonus):
        targets = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x,y] in [3,4]:  # items
                    targets.add((x,y))
        if not targets:
            return None
        return self._move_to_targets_bfs(grid, pos, blocked, targets, danger_soon)

    def _move_to_good_bomb_spot(self, grid, pos, blocked, danger_soon, radius):
        targets = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x,y] == 2:  # box
                    for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nx,ny = x+dx, y+dy
                        if self._is_passable(grid, (nx,ny)) and (nx,ny) not in blocked:
                            targets.add((nx,ny))
        if not targets:
            return None
        return self._move_to_targets_bfs(grid, pos, blocked, targets, danger_soon, max_dist=8)

    def _move_to_targets_bfs(self, grid, start, blocked, targets, danger_soon, max_dist=12):
        q = deque([(start, None, 0)])
        seen = {start}
        while q:
            pos, first_action, dist = q.popleft()
            if pos in targets and first_action is not None:
                return first_action
            if dist >= max_dist:
                continue
            for a in [1,2,3,4]:
                npos = (pos[0] + self.MOVES[a][0], pos[1] + self.MOVES[a][1])
                if npos in seen or not self._is_passable(grid, npos) or npos in blocked or danger_soon[npos]:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action, dist+1))
        return None

    def _safe_explore(self, grid, pos, blocked, danger_soon, valid_moves):
        safe_moves = [a for a in valid_moves if 
                     not danger_soon[self._next_pos(pos, a)]]
        if safe_moves:
            return random.choice(safe_moves)
        return 0

    def _is_passable(self, grid, pos):
        x, y = pos
        return (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 
                grid[x, y] not in [1, 2])

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return (pos[0] + dx, pos[1] + dy)

    def _is_item_near(self, grid, pos, dist=2):
        for x in range(max(0, pos[0]-dist), min(grid.shape[0], pos[0]+dist+1)):
            for y in range(max(0, pos[1]-dist), min(grid.shape[1], pos[1]+dist+1)):
                if grid[x,y] in [3,4]:
                    return True
        return False