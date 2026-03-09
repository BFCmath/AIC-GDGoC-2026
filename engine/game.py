import numpy as np
from .map import Map
from .bomb import Bomb
from .player import Player
# gym?
class BomberEnv:
    N_ACTIONS = 6 # 0: STOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: PLACE_BOMB
    
    def __init__(self, width=13, height=13, max_steps = 100):
        self.width = width 
        self.height = height
        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self.reset()
    
    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.map = Map(self.width, self.height)
        self.players = [
            # ver 1.0: 1v1, top-left & bottom-right
            # TODO: more players, random spawn, etc
            Player(0, 1, 1),
            Player(1, self.width - 2, self.height - 2)
        ]
        
        self.bombs = []
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        bomb_obs = np.zeros((self.width * self.height, 3), dtype=np.int8)
        for i, b in enumerate(self.bombs):
            bomb_obs[i] = [b.x, b.y, b.timer]
        # full obersvability
        return {
            "map": self.map.grid.astype(dtype=np.int8),
            "players": np.array([[p.x, p.y, p.alive, p.bombs_left, p.bomb_radius_bonus] for p in self.players], dtype=np.int8),
            "bombs": bomb_obs
        }
        
    # actions = [player 0 action, player 1 action, ...]
    def step(self, actions):
        self.current_step += 1
        
        for player_id, action in enumerate(actions):
            player = self.players[player_id]
            if not player.alive:
                continue
            
            dx, dy = 0, 0
            if action == 1:
                dx = -1
            elif action == 2:
                dx = 1
            elif action == 3:
                dy = -1
            elif action == 4:
                dy = 1
            elif action == 5:
                if player.bombs_left <= 0:
                    continue
                if any(b.x == player.x and b.y == player.y for b in self.bombs):
                    continue
                new_bomb = Bomb(player.x, player.y, player.id)
                self.bombs.append(new_bomb) # resolve bomb placement also first
                player.bombs_left -= 1
            
            if dx != 0 or dy != 0:
                player.move(dx, dy, self.map.grid, self.players) # move first
                
        new_bombs = []
        for bomb in self.bombs:
            if bomb.step(): # explode second
                self._explode(bomb)
                self.players[bomb.owner_id].bombs_left += 1
            else:
                new_bombs.append(bomb)
        self.bombs = new_bombs
        
        terminated = sum(p.alive for p in self.players) <= 1
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), terminated, truncated
            
    
    def _explode(self, bomb):
        # affected_tiles relate to player.bomb_radius_bonus
        affected_tiles = [(bomb.x, bomb.y)]
        for r in range(0, self.players[bomb.owner_id].bomb_radius_bonus + 1):
            affected_tiles += [
                (bomb.x + r + 1, bomb.y),
                (bomb.x - r - 1, bomb.y),
                (bomb.x, bomb.y + r + 1),
                (bomb.x, bomb.y - r - 1),
            ]
        for tx, ty in affected_tiles:
            if 0 <= tx < self.width and 0 <= ty < self.height:
                # TODO: boxes destroyable, chain explosion
                if self.map._is_wall(tx, ty):
                    continue
                for p in self.players:
                    if p.x == tx and p.y == ty:
                        p.alive = False