class Bomb:
    def __init__(self, x, y, owner_id, timer=3, radius=1):
        self.x = x
        self.y = y
        self.owner_id = owner_id
        self.timer = timer
        self.radius = radius
    
    def step(self):
        self.timer -= 1
        return self.timer <= 0