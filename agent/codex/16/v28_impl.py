"""MetaV26 AdaptiveTempo: pressure in quiet/Apex-like lobbies, near-item recall in high-tempo strong lobbies."""
import importlib.util
from pathlib import Path


def _load(fname, name):
    p = Path(__file__).resolve().parent / fname
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent

Pressure = _load('v2c_impl.py', '_v2c_impl')
NearItem = _load('v25_impl.py', '_v25_impl')

class Agent:
    team_id = 'MetaV28_LobbyAware'
    def __init__(self, agent_id:int):
        self.agent_id = int(agent_id)
        self.turn = 0
        self.pressure = Pressure(agent_id)
        self.nearitem = NearItem(agent_id)
        self.seen_bombs = set()
        self.recent_enemy_bombs = []
        self.total_enemy_bombs = 0
        self.high_tempo_lock = 0
        self.weak_baseline_lock = 0

    def _sync(self, child):
        try:
            child.turn = max(0, self.turn - 1)
        except Exception:
            pass

    def _update_tempo(self, obs):
        bombs = obs.get('bombs', [])
        current = set(); new_enemy = 0; enemy_current = 0
        for b in bombs:
            key = (int(b[0]), int(b[1]), int(b[3]))
            current.add(key)
            if int(b[3]) != self.agent_id:
                enemy_current += 1
                if key not in self.seen_bombs:
                    new_enemy += 1
        self.seen_bombs = current
        self.total_enemy_bombs += new_enemy
        self.recent_enemy_bombs.append(new_enemy)
        if len(self.recent_enemy_bombs) > 32:
            self.recent_enemy_bombs.pop(0)
        return enemy_current, sum(self.recent_enemy_bombs)

    def act(self, obs:dict) -> int:
        self.turn += 1
        try:
            players = obs['players']
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                return 0
            enemy_current, recent = self._update_tempo(obs)
            alive_enemies = sum(1 for i,p in enumerate(players) if i != self.agent_id and int(p[2]) == 1)

            # Tactical/baseline agents tend to dump several bombs extremely early.
            # In that lobby, pure pressure beats item-recall because there are fewer
            # sophisticated box-race opponents and more random kill chances.
            early_baseline_burst = (
                (self.turn <= 18 and enemy_current >= 3) or
                (self.turn <= 22 and self.total_enemy_bombs >= 6)
            )
            if early_baseline_burst:
                self.weak_baseline_lock = max(self.weak_baseline_lock, 120)
            elif self.weak_baseline_lock > 0:
                self.weak_baseline_lock -= 1

            # Strong codex/c7/claude lobbies do not usually burst on turn 4, but
            # they reach a sustained mid-game bomb tempo.  Then use near-item recall
            # to win boxes/items without giving up safety.
            high_tempo = (
                (self.turn >= 24 and enemy_current >= 3) or
                (self.turn >= 28 and self.total_enemy_bombs >= 7) or
                (self.turn >= 45 and recent >= 6) or
                (alive_enemies >= 3 and self.turn >= 80 and recent >= 4)
            )
            if self.weak_baseline_lock <= 0 and high_tempo:
                self.high_tempo_lock = max(self.high_tempo_lock, 45)
            elif self.high_tempo_lock > 0:
                self.high_tempo_lock -= 1

            if self.high_tempo_lock > 0 and self.weak_baseline_lock <= 0:
                self._sync(self.nearitem)
                return int(self.nearitem.act(obs))
            self._sync(self.pressure)
            return int(self.pressure.act(obs))
        except Exception:
            try:
                self._sync(self.pressure)
                return int(self.pressure.act(obs))
            except Exception:
                return 0
