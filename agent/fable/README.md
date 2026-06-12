# Fable agents

## 1.py — FableStatRaceV1 (2026-06-12)

Hybrid search agent designed to consistently beat codex/7, codex/15 and
claude/2 in mixed lobbies. Built from behavior analysis, not opponent
fingerprinting.

### Key analysis findings (scripts/analyze_lobbies.py)
- Nearly all matches between competent agents truncate at step 500; rank is
  decided by the tie-break (kills > boxes > items > bombs).
- 4-Strong lobbies are decided by **boxes**; mixed lobbies by **kills** on the
  weak agents (strong agents almost never kill each other).
- Boxes are finite (~25/map) and fully consumed by ~step 100 → the box race is
  an opening sprint. Items auto-spawn forever → late item hoarding wins the
  3rd tie-break and denies opponents.

### Design pillars
1. **Survival core** (inherited from codex/15): chain-aware bomb danger
   schedule, time-expanded escape BFS, pessimistic anti-trap shields.
2. **Exact stat-race tracking**: reconstructs every player's
   kills/boxes/items/bombs from observation deltas (verified exact vs engine).
   Endgame controller locks in uncatchable leads (safe mode) or pushes the
   precise stat that flips the rank.
3. **Kill harvesting**: guaranteed-kill bombs (enemy escape provably fails),
   kill-share via chain-aligned blasts on doomed enemies, blast saturation on
   behaviorally vulnerable enemies (exposure EWMA of lingering in
   about-to-explode cells).
4. **Trap robustness beyond v15** (fixes v15's own death modes):
   - simultaneous-placement check: own bomb escape must survive a same-tick
     enemy bomb from any adjacent armed enemy;
   - two-level enemy seal model (bomb now = hard veto, bomb after one enemy
     step = heavy penalty), with rational-enemy filter (ignores suicidal seals
     from competent enemies);
   - joint pessimism: all nearby armed enemies bomb the same tick (catches
     two-enemy corridor seals).
5. **Stat efficiency**: box credit timing (skip boxes an enemy bomb pops
   first; chain-aligned bombs share credit), contested-item skip, race modes
   disabled before turn 120 to keep the opening a pure box sprint.

### Benchmarks (scripts/benchmark_lobbies.py)
Cumulative over 1,280 matches on 4 fresh seed bases (80/lobby/base, corner
rotation): fable tops every lobby type —
4S 131 vs v15 105 · 3S1W 141 vs 91 · 2S2Wa 146 vs 128 · 2S2Wb 180 vs 121.
Also #1 TrueSkill in scripts/benchmark_6way_fable.py vs 7/13/15/16/claude2.
Worst-case act() latency 22ms (limit 100ms). Run-to-run variance is real
(hash-seed dependent tie-breaks); judge changes on ≥2 seed bases.
