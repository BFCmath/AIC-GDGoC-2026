import numpy as np

class Map:
    def __init__(self, width=13, height=13):
        self.width=11
        self.height=11
        self.grid = np.zeros((width, height), dtype=int)
        self._setup_walls()
    
    def _setup_walls(self):
        # valid cell from [1, 1] -> [11, 11] (1-index)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
    # 1: wall, 0: grass, 2: box
    def _is_wall(self, x, y):
        return self.grid[x, y] == 1

# TODO: add boxes, map generation, bla bla