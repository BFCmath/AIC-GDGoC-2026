import random
from collections import deque

class Agent:
    team_id = "TacticalTieBreakerV2"
    MOVES = {0:(0,0),1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
    def __init__(self, agent_id:int):
        self.agent_id=int(agent_id)

    def act(self, obs):
        try:
            grid=obs['map']; players=obs['players']; bombs=obs['bombs']
            if self.agent_id>=len(players) or int(players[self.agent_id][2])!=1: return 0
            me=players[self.agent_id]; pos=(int(me[0]),int(me[1])); bombs_left=int(me[3]); radius=1+int(me[4])
            bomb_pos={(int(b[0]),int(b[1])) for b in bombs}
            enemies=[(int(p[0]),int(p[1])) for i,p in enumerate(players) if i!=self.agent_id and int(p[2])==1]
            occupied=set(enemies)
            blocked=set(bomb_pos) | occupied; blocked.discard(pos)
            danger_soon,danger_now=self._danger(grid,bombs,players)
            valid=self._valid(grid,pos,blocked)
            # 1. If in danger, choose move maximizing reachable safe component, not merely nearest safe.
            if pos in danger_now or pos in danger_soon:
                a=self._escape_best(grid,pos,blocked,danger_now,danger_soon)
                if a is not None: return a
                safe=[a for a in valid if self._next(pos,a) not in danger_now]
                return random.choice(safe) if safe else 0
            # 2. Prioritize item if it can be reached safely; capacity early, radius after capacity.
            item_targets=self._item_targets(grid,bombs_left,radius)
            if item_targets:
                a=self._best_target_action(grid,pos,blocked,danger_soon,item_targets,lambda p,d: self._item_score(grid,p,bombs_left,radius,d))
                if a is not None: return a
            # 3. Safe bomb placement. In this game, 500-step survivor tiebreak uses boxes/items/bombs, so safe farming is valuable.
            if bombs_left>0 and pos not in bomb_pos:
                blast=self._blast(grid,pos[0],pos[1],radius)
                boxes=sum(1 for x,y in blast if int(grid[x,y])==2)
                hit=self._can_hit_enemy(grid,pos,enemies,radius)
                trap=self._trap_bonus(grid,pos,enemies,radius)
                if (hit or boxes>=1 or trap>0) and self._can_escape_after_bomb(grid,pos,blocked,danger_soon,radius):
                    # avoid low-value bomb if a nearby item is immediately available
                    near_item=self._nearest_dist(grid,pos,blocked,danger_soon,item_targets,max_depth=3) if item_targets else None
                    if hit or trap>0 or boxes>=2 or near_item is None:
                        return 5
            # 4. Go to best bombing tile, not arbitrary nearest set element.
            spots=self._box_spots(grid,blocked)
            if spots:
                def score(p,d):
                    boxes=sum(1 for x,y in self._blast(grid,p[0],p[1],radius) if int(grid[x,y])==2)
                    # prefer multi-box, reachable, and not a dead-end unless bomb escape exists
                    return 5.0*boxes - 0.4*d + 0.2*self._open_neighbors(grid,p,blocked) - 0.05*(abs(p[0]-6)+abs(p[1]-6))
                a=self._best_target_action(grid,pos,blocked,danger_soon,spots,score,max_depth=14)
                if a is not None: return a
            # 5. Pressure enemies only when we have enough radius/capacity, otherwise keep farming mobility.
            if enemies and (radius>=3 or bombs_left>=2):
                line_spots=self._enemy_line_spots(grid,enemies,radius,blocked)
                if line_spots:
                    a=self._best_target_action(grid,pos,blocked,danger_soon,line_spots,lambda p,d: 4.0-0.3*d,max_depth=12)
                    if a is not None: return a
                a=self._best_target_action(grid,pos,blocked,danger_soon,set(enemies),lambda p,d: 2.0-0.2*d,max_depth=12,allow_target_occupied=True)
                if a is not None: return a
            # 6. Fallback: move to high-mobility safe tile.
            safe=[a for a in valid if self._next(pos,a) not in danger_soon]
            if not safe: return 0
            return max(safe,key=lambda a:self._open_neighbors(grid,self._next(pos,a),blocked)-0.03*(abs(self._next(pos,a)[0]-6)+abs(self._next(pos,a)[1]-6)))
        except Exception:
            return 0

    def _next(self,pos,a):
        dx,dy=self.MOVES.get(a,(0,0)); return (pos[0]+dx,pos[1]+dy)
    def _inb(self,g,x,y): return 0<=x<g.shape[0] and 0<=y<g.shape[1]
    def _passable(self,g,x,y): return self._inb(g,x,y) and int(g[x,y]) in (0,3,4)
    def _valid(self,g,pos,blocked):
        acts=[0]
        for a in [1,2,3,4]:
            np=self._next(pos,a)
            if self._passable(g,*np) and np not in blocked: acts.append(a)
        return acts
    def _blast(self,g,x,y,r):
        tiles={(x,y)}
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            for k in range(1,r+1):
                nx,ny=x+dx*k,y+dy*k
                if not self._inb(g,nx,ny): break
                cell=int(g[nx,ny])
                if cell==1: break
                tiles.add((nx,ny))
                if cell==2: break
        return tiles
    def _danger(self,g,bombs,players,default_radius=2):
        soon=set(); now=set()
        for b in bombs:
            x,y,t,oid=int(b[0]),int(b[1]),int(b[2]),int(b[3])
            if t<=0: continue
            r=1+int(players[oid][4]) if 0<=oid<len(players) else default_radius
            bl=self._blast(g,x,y,r); soon |= bl
            if t<=1: now |= bl
        return soon,now
    def _open_neighbors(self,g,pos,blocked):
        c=0
        for a in [1,2,3,4]:
            np=self._next(pos,a)
            if self._passable(g,*np) and np not in blocked: c+=1
        return c
    def _safe_tiles(self,g,danger):
        return {(x,y) for x in range(g.shape[0]) for y in range(g.shape[1]) if self._passable(g,x,y) and (x,y) not in danger}
    def _escape_best(self,g,start,blocked,danger_now,danger_soon):
        best=None; bs=-10**9
        for a in self._valid(g,start,blocked):
            if a==0: continue
            np=self._next(start,a)
            if np in danger_now: continue
            s=0
            if np not in danger_soon: s+=20
            s+=self._reachable_count(g,np,blocked,danger_soon,depth=8)
            s+=self._open_neighbors(g,np,blocked)
            if s>bs: bs=s; best=a
        return best
    def _reachable_count(self,g,start,blocked,danger,depth=8):
        q=deque([(start,0)]); seen={start}
        while q:
            p,d=q.popleft()
            if d>=depth: continue
            for a in [1,2,3,4]:
                np=self._next(p,a)
                if np in seen or not self._passable(g,*np) or np in blocked or np in danger: continue
                seen.add(np); q.append((np,d+1))
        return len(seen)
    def _move_to_safe(self,g,start,blocked,danger,depth=8):
        q=deque([(start,0,None)]); seen={start}
        while q:
            p,d,first=q.popleft()
            if d>0 and p not in danger: return first
            if d>=depth: continue
            for a in [1,2,3,4,0]:
                np=self._next(p,a)
                if a!=0 and (not self._passable(g,*np) or np in blocked): continue
                if np in seen: continue
                seen.add(np); q.append((np,d+1,a if first is None else first))
        return None
    def _can_escape_after_bomb(self,g,pos,blocked,danger,radius):
        return self._move_to_safe(g,pos,blocked,set(danger)|self._blast(g,pos[0],pos[1],radius),depth=8) is not None
    def _line_clear(self,g,a,b):
        ax,ay=a; bx,by=b
        if ax==bx:
            step=1 if by>ay else -1
            for y in range(ay+step,by,step):
                if int(g[ax,y]) in (1,2): return False
            return True
        if ay==by:
            step=1 if bx>ax else -1
            for x in range(ax+step,bx,step):
                if int(g[x,ay]) in (1,2): return False
            return True
        return False
    def _can_hit_enemy(self,g,pos,enemies,r):
        return any(((pos[0]==e[0] and abs(e[1]-pos[1])<=r) or (pos[1]==e[1] and abs(e[0]-pos[0])<=r)) and self._line_clear(g,pos,e) for e in enemies)
    def _trap_bonus(self,g,pos,enemies,r):
        bl=self._blast(g,pos[0],pos[1],r); val=0
        for e in enemies:
            if e in bl:
                exits=0
                for a in [1,2,3,4]:
                    np=self._next(e,a)
                    if self._passable(g,*np) and np not in bl: exits+=1
                if exits==0: val+=3
                elif exits==1: val+=1
        return val
    def _item_targets(self,g,bombs_left,radius):
        vals=[]
        if bombs_left<=1: vals.append(4)
        if radius<=2: vals.append(3)
        pref={(x,y) for x in range(g.shape[0]) for y in range(g.shape[1]) if int(g[x,y]) in vals}
        if pref: return pref
        return {(x,y) for x in range(g.shape[0]) for y in range(g.shape[1]) if int(g[x,y]) in (3,4)}
    def _item_score(self,g,p,bombs_left,radius,d):
        c=int(g[p[0],p[1]])
        s=8.0-0.45*d
        if c==4 and bombs_left<=1: s+=4
        if c==3 and radius<=2: s+=3
        return s
    def _best_target_action(self,g,start,blocked,danger,targets,score_fn,max_depth=16,allow_target_occupied=False):
        if not targets: return None
        q=deque([(start,0,None)]); seen={start}; best=None; bs=-10**9
        while q:
            p,d,first=q.popleft()
            if d>0 and p in targets:
                s=score_fn(p,d)
                if s>bs: bs=s; best=first
            if d>=max_depth: continue
            for a in [1,2,3,4]:
                np=self._next(p,a)
                if np in seen or not self._passable(g,*np): continue
                if np in blocked and not (allow_target_occupied and np in targets): continue
                if np in danger: continue
                seen.add(np); q.append((np,d+1,a if first is None else first))
        return best
    def _nearest_dist(self,g,start,blocked,danger,targets,max_depth=8):
        q=deque([(start,0)]); seen={start}
        while q:
            p,d=q.popleft()
            if d>0 and p in targets: return d
            if d>=max_depth: continue
            for a in [1,2,3,4]:
                np=self._next(p,a)
                if np in seen or not self._passable(g,*np) or np in blocked or np in danger: continue
                seen.add(np); q.append((np,d+1))
        return None
    def _box_spots(self,g,blocked):
        spots=set()
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                if int(g[x,y])!=2: continue
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    p=(x+dx,y+dy)
                    if self._passable(g,*p) and p not in blocked: spots.add(p)
        return spots
    def _enemy_line_spots(self,g,enemies,r,blocked):
        spots=set()
        for ex,ey in enemies:
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                for k in range(1,r+1):
                    p=(ex+dx*k,ey+dy*k)
                    if not self._inb(g,*p) or int(g[p[0],p[1]]) in (1,2): break
                    if self._passable(g,*p) and p not in blocked: spots.add(p)
        return spots
