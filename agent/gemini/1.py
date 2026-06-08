import random
from collections import deque
import time
import numpy as np

class CalculatedSurvivorAgent:
    MOVES = {
        0: (0, 0),   # STOP
        1: (-1, 0),  # LEFT
        2: (1, 0),   # RIGHT
        3: (0, -1),  # UP
        4: (0, 1),   # DOWN
    }
    team_id = "CalculatedSurvivor"

    def __init__(self, agent_id):
        self.agent_id = int(agent_id)
        self.step_count = 0
        self.INF = 999

    def act(self, observation):
        self.step_count += 1
        
        grid = observation["map"]
        players = observation["players"]
        bombs = observation["bombs"].tolist() if isinstance(observation["bombs"], np.ndarray) else list(observation["bombs"])

        # Nếu mình đã chết, nằm im
        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        my_radius = max(1, int(bomb_bonus) + 1)
        
        # Xác định đối thủ còn sống
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]
        enemy_positions = set(enemies)
        
        # Xác định vật cản vật lý (không thể bước vào)
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        physical_blocks = enemy_positions | bomb_positions
        physical_blocks.discard(my_pos) # Tránh tự block mình nếu vừa đặt bom

        # =====================================================================
        # LỚP 1: ABSOLUTE VETO - TÍNH TOÁN BẢN ĐỒ TTE (TIME-TO-EXPLOSION)
        # =====================================================================
        tte_map = self._build_tte_map(grid, bombs, players)

        # Lọc các hành động an toàn vật lý
        valid_actions = [0]
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(my_pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in physical_blocks:
                valid_actions.append(a)

        # VETO: Chỉ giữ lại các hành động có TỐI THIỂU 1 đường thoát thân an toàn
        survivable_actions = []
        for a in valid_actions:
            nx, ny = self._next_pos(my_pos, a)
            # Kiểm tra xem từ (nx, ny) ở step 1, có chạy thoát được bom không?
            if self._can_survive_from(grid, nx, ny, tte_map, physical_blocks, start_time=1):
                survivable_actions.append(a)

        # Nếu không có lối thoát, chọn hành động giúp sống lâu nhất (delay cái chết)
        if not survivable_actions:
            return self._get_longest_survival_move(my_pos, valid_actions, tte_map)

        # =====================================================================
        # KIỂM TRA ĐIỀU KIỆN ĐẶT BOM (GHOST BOMB / FARM BOMB)
        # =====================================================================
        can_place_bomb = False
        if bombs_left > 0 and my_pos not in bomb_positions:
            # Tưởng tượng đặt 1 quả bom ở đây, cập nhật TTE map
            imagined_bombs = bombs + [[my_x, my_y, 7, self.agent_id]]
            imagined_tte = self._build_tte_map(grid, imagined_bombs, players)
            # Nếu đặt xong vẫn có đường lui (ở action STOP hoặc đi chỗ khác)
            if self._can_survive_from(grid, my_pos[0], my_pos[1], imagined_tte, physical_blocks, start_time=0):
                can_place_bomb = True

        # =====================================================================
        # LỚP 2: STATE MACHINE THEO GIAI ĐOẠN (FARM -> PARASITE -> ENDGAME)
        # =====================================================================
        
        # Lấy khoảng cách đến đối thủ gần nhất
        dist_to_closest_enemy = min([self._manhattan(my_pos, e) for e in enemies]) if enemies else self.INF

        # PHASE 1: FARMING (Step 0 - 150)
        if self.step_count < 150:
            action = self._farming_phase(grid, my_pos, my_radius, survivable_actions, can_place_bomb, physical_blocks, tte_map)
            if action is not None:
                return action

        # PHASE 2 & 3: PARASITE / HIDING / TIE-BREAKER (Step 150 - 500)
        else:
            # Nhặt item an toàn mồ côi nếu ở gần (bán kính 3 ô)
            item_action = self._safe_item_pickup(grid, my_pos, survivable_actions, physical_blocks, tte_map, radius=3)
            if item_action is not None:
                return item_action

            # Chế độ trốn chạy: Tìm ô an toàn xa đối thủ nhất
            hide_action = self._hide_from_enemies(grid, my_pos, enemies, survivable_actions, physical_blocks, tte_map)
            
            # Kích hoạt Ghost Bombing (Bom vô nghĩa) để cày Tie-breaker nếu quá rảnh
            if hide_action == 0 and can_place_bomb and dist_to_closest_enemy > 4 and self.step_count > 300:
                return 5 # Đặt bom cày chỉ số
                
            if hide_action is not None:
                return hide_action

        # Fallback: Random một hành động an toàn
        return random.choice(survivable_actions) if survivable_actions else 0

    # =====================================================================
    # CORE ENGINE: TTE (TIME-TO-EXPLOSION) VÀ CHAIN REACTIONS
    # =====================================================================
    def _build_tte_map(self, grid, bombs, players):
        w, h = grid.shape
        tte = { (x, y): self.INF for x in range(w) for y in range(h) }
        
        if len(bombs) == 0:
            return tte

        # Phân tích bom
        bomb_info = {}
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner = int(b[3]) if len(b) > 3 else -1
            radius = max(1, int(players[owner][4]) + 1) if 0 <= owner < len(players) else 2
            bomb_info[(bx, by)] = {'timer': timer, 'radius': radius}

        # Xử lý bom dây chuyền (Chain Reaction)
        changed = True
        while changed:
            changed = False
            for (x1, y1), b1 in bomb_info.items():
                blast_tiles = self._blast_tiles(grid, x1, y1, b1['radius'])
                for (x2, y2) in blast_tiles:
                    if (x2, y2) in bomb_info and (x1, y1) != (x2, y2):
                        # Nếu bom 1 nổ trước bom 2, bom 2 sẽ nổ theo ngay lập tức
                        if b1['timer'] < bomb_info[(x2, y2)]['timer']:
                            bomb_info[(x2, y2)]['timer'] = b1['timer']
                            changed = True

        # Render vụ nổ lên bản đồ TTE
        for (bx, by), b in bomb_info.items():
            blast_tiles = self._blast_tiles(grid, bx, by, b['radius'])
            for (x, y) in blast_tiles:
                if b['timer'] < tte[(x, y)]:
                    tte[(x, y)] = b['timer']

        return tte

    def _can_survive_from(self, grid, start_x, start_y, tte_map, physical_blocks, start_time):
        """BFS tìm xem từ vị trí này có đường nào sống sót qua khỏi các vụ nổ không"""
        if tte_map[(start_x, start_y)] <= start_time:
            return False

        q = deque([((start_x, start_y), start_time)])
        seen = {((start_x, start_y), start_time)}
        max_tte_in_map = max([t for t in tte_map.values() if t != self.INF] + [0])
        
        while q:
            (x, y), t = q.popleft()
            
            # Nếu đã sống qua được thời điểm quả bom nổ lâu nhất, tức là an toàn tuyệt đối
            if tte_map[(x, y)] == self.INF and t > max_tte_in_map:
                return True
                
            next_t = t + 1
            # Thử đứng yên hoặc di chuyển
            for a in [0, 1, 2, 3, 4]:
                nx, ny = self._next_pos((x, y), a)
                if not self._passable(grid, nx, ny): continue
                if a != 0 and (nx, ny) in physical_blocks: continue
                
                # Chết chìm trong vụ nổ
                if tte_map[(nx, ny)] <= next_t:
                    continue
                    
                state = ((nx, ny), next_t)
                if state not in seen:
                    # Tránh tràn RAM / Timeout bằng cách cắt nhánh nếu t quá lớn
                    if next_t > 15: 
                        return True
                    seen.add(state)
                    q.append(state)
                    
        return False

    def _get_longest_survival_move(self, my_pos, valid_actions, tte_map):
        """Khi chắc chắn chết, chọn lối đi kéo dài sự sống nhất (tránh chết ngay step 1)"""
        best_action = 0
        max_time = -1
        for a in valid_actions:
            nx, ny = self._next_pos(my_pos, a)
            time_alive = tte_map.get((nx, ny), 0)
            if time_alive > max_time:
                max_time = time_alive
                best_action = a
        return best_action

    # =====================================================================
    # TACTICAL PHASES
    # =====================================================================
    def _farming_phase(self, grid, my_pos, my_radius, survivable_actions, can_place_bomb, physical_blocks, tte_map):
        # 1. Nếu đang đứng cạnh hộp và có thể đặt bom an toàn -> Xúc
        if can_place_bomb:
            boxes_hit = self._count_boxes_in_blast(grid, my_pos[0], my_pos[1], my_radius)
            if boxes_hit > 0:
                return 5

        # 2. Tìm Item an toàn
        item_action = self._safe_item_pickup(grid, my_pos, survivable_actions, physical_blocks, tte_map, radius=13)
        if item_action is not None:
            return item_action

        # 3. Tìm góc đặt bom cày hộp
        targets = self._box_bomb_spots(grid, physical_blocks)
        move = self._move_to_safest_target(grid, my_pos, targets, survivable_actions, physical_blocks, tte_map)
        if move is not None:
            return move
            
        return None

    def _safe_item_pickup(self, grid, my_pos, survivable_actions, physical_blocks, tte_map, radius=13):
        item_tiles = {(x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1]) if grid[x, y] in [3, 4] and self._manhattan(my_pos, (x, y)) <= radius}
        if item_tiles:
            return self._move_to_safest_target(grid, my_pos, item_tiles, survivable_actions, physical_blocks, tte_map)
        return None

    def _hide_from_enemies(self, grid, my_pos, enemies, survivable_actions, physical_blocks, tte_map):
        if not enemies:
            return 0
            
        best_action = 0
        max_min_dist = -1
        
        # Đánh giá điểm đến của từng action
        for a in survivable_actions:
            nx, ny = self._next_pos(my_pos, a)
            # Tính min distance từ ô này đến mọi đối thủ
            min_dist = min([self._manhattan((nx, ny), e) for e in enemies])
            
            # Thưởng thêm nếu ô tiếp theo có TTE = vô cực (an toàn 100%)
            score = min_dist
            if tte_map[(nx, ny)] == self.INF:
                score += 10
                
            if score > max_min_dist:
                max_min_dist = score
                best_action = a
                
        # Nếu đã cách xa >= 6 ô và đứng yên an toàn, thì đừng nhúc nhích tốn công
        if best_action != 0 and max_min_dist < 16 and 0 in survivable_actions:
            min_dist_current = min([self._manhattan(my_pos, e) for e in enemies])
            if min_dist_current >= 6 and tte_map[my_pos] == self.INF:
                return 0
                
        return best_action

    # =====================================================================
    # PATHFINDING & UTILS
    # =====================================================================
    def _move_to_safest_target(self, grid, start, targets, survivable_actions, physical_blocks, tte_map):
        if not targets: return None
        if start in targets and 0 in survivable_actions: return 0
        
        q = deque([(start, None)])
        seen = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in targets and first_action is not None:
                # Phải đảm bảo action đầu tiên nằm trong list VETO passed
                if first_action in survivable_actions:
                    return first_action
                    
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen: continue
                if not self._passable(grid, nx, ny): continue
                if npos in physical_blocks: continue
                # Tránh đi qua các ô đang sắp nổ (heuristics nhanh)
                if tte_map[npos] < 5: continue 
                
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))
        return None

    def _box_bomb_spots(self, grid, occupied):
        targets = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 2: continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in occupied:
                        targets.add((nx, ny))
        return targets

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y): break
                cell = grid[x, y]
                if cell == 1: break # Tường cứng cản bom
                tiles.add((x, y))
                if cell == 2: break # Hộp cản bom nhưng vỡ
        return tiles

    def _count_boxes_in_blast(self, grid, bx, by, radius):
        return sum(1 for x, y in self._blast_tiles(grid, bx, by, radius) if grid[x, y] == 2)

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and grid[x, y] in [0, 3, 4]

    def _manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])