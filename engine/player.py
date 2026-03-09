import numpy as np

class Player:
    def __init__(self, player_id, x, y):
        self.id = player_id
        self.x = x
        self.y = y
        self.alive = True
        self.bombs_left = 1
        self.bomb_radius_bonus = 0
    
    def move(self, dx, dy, grid, players):
        new_x = self.x + dx
        new_y = self.y + dy
    
        if not (0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]):
            return
    
        if grid[new_x, new_y] == 1:
            return
        # allow overlap
        # for p in players:
        #     if p.id != self.id and p.alive and p.x == new_x and p.y == new_y:
        #         return
        
        self.x = new_x
        self.y = new_y