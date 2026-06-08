import numpy as np
from collections import deque, defaultdict
import math

class ShadowAdaptiveBomber:
    team_id = "ShadowAdaptiveBomber"

    # Hành động
    STOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    BOMB = 5
    MOVES = {STOP: (0, 0), LEFT: (-1, 0), RIGHT: (1, 0), UP: (0, -1), DOWN: (0, 1)}
    ACTION_ORDER = [STOP, LEFT, RIGHT, UP, DOWN]  # không gồm BOMB, BOMB xét riêng

    def __init__(self, agent_id):
        self.agent_id = int(agent_id)
        # Bộ nhớ bom: (x, y) -> (timer, radius, owner_id)
        self.bomb_memory = {}
        # Lịch sử đối thủ: id -> list of dict với info mỗi step (giới hạn 30 step gần nhất)
        self.opp_history = defaultdict(lambda: deque(maxlen=30))
        self.current_mode = "farmer"   # farmer / hunter / survivor
        self.step_count = 0
        # Lưu step trước để tính delta cho phân tích đối thủ
        self.last_alive = set()
        self.last_positions = {}

    def act(self, observation):
        self.step_count += 1
        grid = observation["map"]          # numpy 13x13, 0: cỏ, 1: tường, 2: hộp, 3: bán kính, 4: capacity
        players = observation["players"]   # list: [x, y, alive, bombs_left, bomb_bonus]
        bombs_raw = observation["bombs"]   # list: [x, y, timer, owner_id]

        # ----- 1. Tiền xử lý thông tin -----
        alive_players = {i: p for i, p in enumerate(players) if p[2] == 1}
        if self.agent_id not in alive_players:
            return self.STOP

        my_x, my_y, _, my_bombs_left, my_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        my_radius = max(1, int(my_bonus) + 1)

        # Cập nhật bộ nhớ bom (bán kính cố định)
        current_bomb_set = set()
        for b in bombs_raw:
            bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            pos = (bx, by)
            current_bomb_set.add(pos)
            if pos not in self.bomb_memory:
                # Bom mới: lấy bán kính từ người đặt (tại thời điểm đặt là hiện tại)
                if 0 <= owner < len(players) and players[owner][2] == 1:
                    radius = max(1, int(players[owner][4]) + 1)
                else:
                    radius = 2  # fallback an toàn
                self.bomb_memory[pos] = [timer, radius, owner]
            else:
                # Cập nhật timer, giữ nguyên bán kính đã lưu
                self.bomb_memory[pos][0] = timer
        # Xoá bom đã nổ (không còn trong danh sách)
        for pos in list(self.bomb_memory.keys()):
            if pos not in current_bomb_set:
                del self.bomb_memory[pos]

        # Cập nhật lịch sử đối thủ
        for opp_id, p in alive_players.items():
            if opp_id == self.agent_id:
                continue
            ox, oy = int(p[0]), int(p[1])
            bombs_left = int(p[3])
            bonus = int(p[4])
            # Nếu có vị trí trước đó, tính hành động (đơn giản: hướng di chuyển)
            action_taken = None
            if opp_id in self.last_positions and self.last_positions[opp_id] is not None:
                prev_pos = self.last_positions[opp_id]
                if ox != prev_pos[0] or oy != prev_pos[1]:
                    dx, dy = ox - prev_pos[0], oy - prev_pos[1]
                    if dx == -1: action_taken = self.LEFT
                    elif dx == 1: action_taken = self.RIGHT
                    elif dy == -1: action_taken = self.UP
                    elif dy == 1: action_taken = self.DOWN
                else:
                    action_taken = self.STOP
            # Ghi nhận
            self.opp_history[opp_id].append({
                "pos": (ox, oy),
                "bombs_left": bombs_left,
                "bonus": bonus,
                "action": action_taken
            })
            self.last_positions[opp_id] = (ox, oy)
        # Xoá history của đối thủ đã chết
        for opp_id in list(self.opp_history.keys()):
            if opp_id not in alive_players:
                del self.opp_history[opp_id]
                if opp_id in self.last_positions:
                    del self.last_positions[opp_id]

        # ----- 2. Phân tích đối thủ -> chọn chế độ -----
        enemy_ids = [i for i in alive_players if i != self.agent_id]
        mode = self._select_mode(enemy_ids, alive_players, grid)

        # ----- 3. Lập bản đồ nguy hiểm & ô an toàn -----
        danger_now, danger_soon = self._compute_danger(grid, players)
        safe_tiles = self._safe_tiles(grid, danger_soon)

        # ----- 4. Lấy danh sách ô bị chiếm (đối thủ + bom) -----
        occupied = set()
        for i, p in alive_players.items():
            if i != self.agent_id:
                occupied.add((int(p[0]), int(p[1])))
        for bpos in self.bomb_memory:
            if bpos != my_pos:   # cho phép đứng trên ô mình vừa đặt bom (sẽ thoát sau)
                occupied.add(bpos)

        # ----- 5. Sinh hành động theo chế độ -----
        action = self._action_for_mode(mode, grid, my_pos, my_bombs_left, my_radius,
                                       danger_now, danger_soon, safe_tiles, occupied,
                                       alive_players, enemy_ids)
        return action

    # --------------- CÁC HÀM HỖ TRỢ ---------------

    def _select_mode(self, enemy_ids, alive_players, grid):
        """Meta‑policy: chọn mode dựa trên phân tích đối thủ và tình huống."""
        if not enemy_ids:
            return "farmer"   # còn mỗi mình thì farm thoải mái

        # Phân loại từng đối thủ (đơn giản hoá)
        aggressives = 0
        campers = 0
        for eid in enemy_ids:
            history = self.opp_history[eid]
            if len(history) < 15:
                # Chưa đủ dữ liệu -> mặc định coi là thường
                continue
            bombs_placed = sum(1 for h in history if h["action"] is None and h["bombs_left"] < self._prev_bombs(history, -1))
            # Cách đo đơn giản: tần suất đặt bom khi có kẻ thù gần
            # Nhưng để tiết kiệm, ta dùng heuristic: nếu bomb_placed nhiều và khoảng cách đến đối thủ nhỏ -> hung hăng
            moves_count = sum(1 for h in history if h["action"] is not None and h["action"] != self.STOP)
            if moves_count > 0:
                # Nếu thường xuyên đặt bom và có xu hướng tiến về phía đối thủ -> aggressive
                # Tạm dùng: nếu trong 10 step gần có >=2 lần đặt bom, coi là aggressive
                recent_bombs = sum(1 for h in list(history)[-10:] if h["action"] is None and h["bombs_left"] < self._prev_bombs(list(history)[-10:], -1))
                if recent_bombs >= 2:
                    aggressives += 1
                else:
                    # Nếu thường đứng yên hoặc quanh quẩn góc -> camper (có thể né tránh)
                    unique_pos = set(h["pos"] for h in history)
                    if len(unique_pos) <= 3:
                        campers += 1

        # Tự đánh giá bản thân
        my_bonus = int(alive_players[self.agent_id][4])
        my_radius = max(1, my_bonus + 1)

        # Logic chọn mode:
        # - Nếu đối thủ phần lớn hung hăng và ta yếu hơn -> survivor
        # - Nếu ta mạnh và có nhiều đối thủ hung hăng -> hunter (đặt bẫy)
        # - Còn lại -> farmer (an toàn, tích luỹ)
        if aggressives >= len(enemy_ids) * 0.6:
            if my_radius <= 2:   # còn yếu
                return "survivor"
            else:
                return "hunter"
        elif campers >= len(enemy_ids) * 0.5:
            # Nhiều camper -> farmer để phá hộp, câu giờ
            return "farmer"
        # Trường hợp hỗn hợp: dựa vào thế trận
        if self.step_count > 350:  # gần cuối -> ưu tiên sinh tồn để tie‑break
            return "survivor"
        return "farmer"

    def _prev_bombs(self, history_list, idx):
        """Lấy bombs_left của bước trước đó trong history (phục vụ phát hiện đặt bom)."""
        # Đơn giản: không implement sâu, chỉ cần heuristic
        return 0

    def _compute_danger(self, grid, players):
        """Tạo danger_now (sẽ nổ ở step này) và danger_soon (sẽ nổ trong vài bước)."""
        danger_now = set()
        danger_soon = set()
        for pos, (timer, radius, owner) in self.bomb_memory.items():
            if timer <= 0:
                continue
            blast = self._blast_tiles(grid, pos[0], pos[1], radius)
            danger_soon.update(blast)
            if timer <= 1:
                danger_now.update(blast)
        return danger_now, danger_soon

    def _blast_tiles(self, grid, bx, by, radius):
        """Ô bị ảnh hưởng bởi bom tại (bx,by) với bán kính cho trước."""
        tiles = {(bx, by)}
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            for r in range(1, radius+1):
                nx, ny = bx + dx*r, by + dy*r
                if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                    break
                cell = grid[nx, ny]
                if cell == 1:   # tường
                    break
                tiles.add((nx, ny))
                if cell == 2:   # hộp chặn
                    break
        return tiles

    def _safe_tiles(self, grid, danger_soon):
        safe = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x,y] in [0,3,4] and (x,y) not in danger_soon:
                    safe.add((x,y))
        return safe

    def _action_for_mode(self, mode, grid, my_pos, bombs_left, radius,
                         danger_now, danger_soon, safe_tiles, occupied, alive_players, enemy_ids):
        # --- 1. Nếu đang đứng trong danger_now, ưu tiên thoát ngay ---
        if my_pos in danger_now:
            move = self._move_to_safe(grid, my_pos, occupied, danger_now, safe_tiles)
            if move is not None:
                return move
            # fallback: chọn bất kỳ nước đi không vào danger_now
            for act in self.ACTION_ORDER:
                nx, ny = self._next_pos(my_pos, act)
                if self._passable(grid, nx, ny) and (nx,ny) not in occupied and (nx,ny) not in danger_now:
                    return act
            return self.STOP

        # --- 2. Nếu đang trong vùng danger_soon, tìm đường ra an toàn ---
        if my_pos in danger_soon:
            move = self._move_to_safe(grid, my_pos, occupied, danger_soon, safe_tiles)
            if move is not None:
                return move

        # --- 3. Hành vi đặc thù theo mode ---
        if mode == "survivor":
            # Chỉ tập trung sống sót: tránh mọi rủi ro, nhặt đồ nếu cực an toàn
            # Ưu tiên di chuyển đến ô an toàn gần nhất nếu đang bị đe dọa
            if my_pos in danger_soon:
                move = self._move_to_safe(grid, my_pos, occupied, danger_soon, safe_tiles)
                if move is not None:
                    return move
            # Nhặt item nhưng chỉ khi thật an toàn
            item_move = self._move_to_items(grid, my_pos, occupied, danger_soon, prefer=(bombs_left<=1, radius<=2))
            if item_move is not None:
                return item_move
            # Không đặt bom, chỉ di chuyển đến vùng an toàn nhiều nhất
            best_move = self._move_to_most_promising_safe(grid, my_pos, occupied, safe_tiles)
            return best_move if best_move is not None else self.STOP

        elif mode == "farmer":
            # Phá hộp, nhặt đồ, đặt bom an toàn
            # 1. Nhặt item
            item_move = self._move_to_items(grid, my_pos, occupied, danger_soon, prefer=(bombs_left<=1, radius<=2))
            if item_move is not None:
                return item_move
            # 2. Đặt bom phá hộp nếu có đường thoát
            if bombs_left > 0 and my_pos not in (p for p in self.bomb_memory):
                boxes_hit = self._count_boxes_in_blast(grid, my_pos, radius)
                if boxes_hit > 0:
                    # Kiểm tra thoát được không
                    my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
                    combined = danger_now.union(my_blast)
                    escape = self._move_to_safe(grid, my_pos, occupied, combined, safe_tiles)
                    if escape is not None:
                        return self.BOMB
            # 3. Di chuyển đến vị trí đặt bom tiềm năng (cạnh hộp)
            box_bomb_spot = self._find_box_bomb_spot(grid, my_pos, occupied, danger_soon, bombs_left)
            if box_bomb_spot is not None:
                return box_bomb_spot
            # 4. Di chuyển an toàn vào vùng nhiều hộp
            explore_move = self._move_to_unexplored_safe(grid, my_pos, occupied, safe_tiles)
            return explore_move if explore_move is not None else self.STOP

        elif mode == "hunter":
            # Tìm diệt đối thủ: đặt bẫy hoặc áp sát
            # 1. Nếu có cơ hội kill an toàn (đối thủ đứng cạnh hoặc vào bẫy)
            kill_action = self._try_kill_opponent(grid, my_pos, radius, bombs_left, occupied, danger_now, danger_soon, alive_players, enemy_ids)
            if kill_action is not None:
                return kill_action
            # 2. Áp sát đối thủ nhưng an toàn
            chase_move = self._chase_enemy(grid, my_pos, occupied, danger_soon, enemy_ids, alive_players)
            if chase_move is not None:
                return chase_move
            # 3. Nếu không, quay về farmer
            item_move = self._move_to_items(grid, my_pos, occupied, danger_soon, prefer=(bombs_left<=1, radius<=2))
            if item_move is not None:
                return item_move
            safe_move = self._move_to_most_promising_safe(grid, my_pos, occupied, safe_tiles)
            return safe_move if safe_move is not None else self.STOP

        # Mặc định phòng thủ
        return self.STOP

    # --- Các hàm di chuyển & tìm đường ---
    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0]+dx, pos[1]+dy

    def _passable(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] in [0,3,4]

    def _move_to_safe(self, grid, start, occupied, danger_set, safe_tiles):
        """BFS tìm đường đến ô an toàn (không trong danger_set), tránh occupied và danger_set."""
        q = deque([(start, None)])
        seen = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in safe_tiles and pos != start:
                return first_action
            for act in self.ACTION_ORDER:
                nx, ny = self._next_pos(pos, act)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in occupied or npos in danger_set:
                    continue
                seen.add(npos)
                q.append((npos, act if first_action is None else first_action))
        return None

    def _move_to_items(self, grid, start, occupied, danger_soon, prefer):
        """BFS tìm đường đến item (3,4), ưu tiên loại đang cần."""
        prefer_radius, prefer_capacity = prefer
        item_cells = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                val = grid[x, y]
                if val == 3:
                    if prefer_radius:
                        item_cells.add((x, y))
                elif val == 4:
                    if prefer_capacity:
                        item_cells.add((x, y))
        if not item_cells:
            # lấy tất cả item
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    if grid[x, y] in [3,4]:
                        item_cells.add((x,y))
        if not item_cells:
            return None
        return self._bfs_to_targets(grid, start, occupied, danger_soon, item_cells)

    def _bfs_to_targets(self, grid, start, occupied, danger_soon, targets):
        q = deque([(start, None)])
        seen = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in targets:
                return first_action
            for act in self.ACTION_ORDER:
                nx, ny = self._next_pos(pos, act)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in occupied or npos in danger_soon:
                    continue
                seen.add(npos)
                q.append((npos, act if first_action is None else first_action))
        return None

    def _count_boxes_in_blast(self, grid, pos, radius):
        cnt = 0
        for x, y in self._blast_tiles(grid, pos[0], pos[1], radius):
            if grid[x, y] == 2:
                cnt += 1
        return cnt

    def _find_box_bomb_spot(self, grid, start, occupied, danger_soon, bombs_left):
        """Tìm ô trống kề hộp để đặt bom, và đường đi tới đó an toàn."""
        if bombs_left == 0:
            return None
        # Tạo tập ô đặt bom tiềm năng: ô trống kề ít nhất 1 hộp
        candidates = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] == 2:  # hộp
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x+dx, y+dy
                        if self._passable(grid, nx, ny) and (nx,ny) not in occupied:
                            candidates.add((nx, ny))
        if not candidates:
            return None
        return self._bfs_to_targets(grid, start, occupied, danger_soon, candidates)

    def _move_to_most_promising_safe(self, grid, start, occupied, safe_tiles):
        """Di chuyển đến ô an toàn gần nhất, nếu không có thì đứng yên."""
        if start in safe_tiles and all(True for _ in []): # đơn giản: di chuyển đến bất kỳ ô an toàn nào
            return self._bfs_to_targets(grid, start, occupied, set(), safe_tiles)
        return self.STOP

    def _move_to_unexplored_safe(self, grid, start, occupied, safe_tiles):
        """Di chuyển về phía vùng còn hộp (ô 2) để phá."""
        box_positions = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x,y] == 2:
                    box_positions.add((x,y))
        if not box_positions:
            return self._move_to_most_promising_safe(grid, start, occupied, safe_tiles)
        # BFS đến ô an toàn gần hộp nhất
        return self._bfs_to_targets_priority(grid, start, occupied, set(), box_positions, safe_tiles)

    def _bfs_to_targets_priority(self, grid, start, occupied, danger_soon, targets, safe_zone):
        """BFS ưu tiên đến ô trong safe_zone mà gần targets (dùng heuristic đơn giản)."""
        q = deque([(start, None)])
        seen = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in safe_zone:
                # Tìm thấy ô an toàn, ưu tiên ô gần target
                # Ở đây ta chỉ cần trả về first_action đầu tiên đến được safe_zone
                return first_action
            for act in self.ACTION_ORDER:
                nx, ny = self._next_pos(pos, act)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in occupied or npos in danger_soon:
                    continue
                seen.add(npos)
                q.append((npos, act if first_action is None else first_action))
        return None

    def _try_kill_opponent(self, grid, my_pos, radius, bombs_left, occupied, danger_now, danger_soon, alive_players, enemy_ids):
        """Cố gắng tiêu diệt đối thủ bằng cách đặt bom hoặc dụ."""
        if bombs_left == 0:
            return None
        # Kiểm tra xem có đối thủ nào đứng trong phạm vi nổ nếu ta đặt bom tại vị trí hiện tại
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
        for eid in enemy_ids:
            epos = (int(alive_players[eid][0]), int(alive_players[eid][1]))
            if epos in my_blast:
                # Phải chắc chắn ta thoát được
                combined = danger_now.union(my_blast)
                escape = self._move_to_safe(grid, my_pos, occupied, combined, set())
                if escape is not None:
                    return self.BOMB
        # Nếu đối thủ ở cạnh và ta có thể đặt bom rồi chạy
        for eid in enemy_ids:
            epos = (int(alive_players[eid][0]), int(alive_players[eid][1]))
            if abs(epos[0]-my_pos[0]) + abs(epos[1]-my_pos[1]) == 1:
                return self.BOMB  # đặt bom ngay cạnh rồi thoát (escape kiểm tra sau)
        return None

    def _chase_enemy(self, grid, my_pos, occupied, danger_soon, enemy_ids, alive_players):
        """Tiến về phía đối thủ gần nhất nhưng vẫn an toàn."""
        if not enemy_ids:
            return None
        target = None
        min_dist = 999
        for eid in enemy_ids:
            epos = (int(alive_players[eid][0]), int(alive_players[eid][1]))
            dist = abs(epos[0]-my_pos[0]) + abs(epos[1]-my_pos[1])
            if dist < min_dist:
                min_dist = dist
                target = epos
        if target is None:
            return None
        return self._bfs_to_targets(grid, my_pos, occupied, danger_soon, {target})

    # -------------------------------------------------

# Convenience alias
Agent = ShadowAdaptiveBomber