Now I have a thorough understanding of the codebase. Let me create a comprehensive RL training strategy document.Now I have everything I need to create a comprehensive, visually rich RL training strategy.Here's the full deep-learning training strategy for Bomberland, grounded in what your rule-based agents (1.py / 2.py / 4.py) already know.

---

## Strategy overview

Your rule-based agents are goldmines — they encode hard-won knowledge about danger schedules, escape BFS, trap scoring, and item priority. The RL strategy is to **start from their behavior, then transcend their limitations** (fixed heuristic weights, no opponent modeling, no long-horizon planning).

---

## Phase 1 — Behaviour Cloning (BC warmup, ~1–2 days)

Run your existing rule agents on thousands of self-play games, record `(obs, action)` pairs, and train a supervised CNN+LSTM to imitate them. This gives your RL agent a sane starting point — it already knows how to escape bombs and farm boxes before it sees a single reward signal.

```python
# Observation encoding — 13×13 multi-channel tensor
channels = {
    0: "walls",           # grid == 1
    1: "boxes",           # grid == 2
    2: "empty",           # grid == 0
    3: "radius_item",     # grid == 3
    4: "capacity_item",   # grid == 4
    5: "self_position",   # one-hot
    6: "enemy_positions", # all enemies
    7: "bomb_locs",       # active bombs
    8..14: "danger_t1..7" # blast coverage per timestep (your _danger_schedule output)
}
# + scalar head: [my_radius, bombs_left, capacity, step/500, enemies_alive]
```

BC loss is just cross-entropy over the 6 actions. Train until accuracy ~65–75% — beyond that you're overfitting to the teacher's suboptimalities.

---

## Phase 2 — DQN with self-play and PER

Switch to RL using a **Dueling Double DQN** with Prioritized Experience Replay. The BC checkpoint is the starting weights. Key design decisions:

**Network:** `Conv2d(15, 64, 3) → Conv2d(64, 64, 3) → flatten → concat(scalars) → LSTM(256) → [Value head | Advantage head]`

**Replay buffer:** Prioritize transitions where the agent died (rare, high-signal). A death from a bomb that the rule-based teacher would have avoided is exactly the transition to learn from.

**Self-play setup:** Maintain a snapshot pool of your agent at checkpoints every N steps. Your opponents are sampled exactly like the BTC arena: 40% near-rating, 30% top snapshot, 30% random. This forces the agent to be robust, not just exploit one strategy.

**Exploration:** Use ε-greedy decayed from 0.3 to 0.05, but also add a small bonus for visiting under-explored map tiles early in training to encourage box farming.

---

## Phase 3 — PPO + shaped rewards (main training loop)

Switch from DQN to **Proximal Policy Optimization** once your DQN policy has converged. PPO is better suited for multi-agent environments because:
- The clipped objective prevents catastrophic forgetting when opponents change
- You can run many parallel environments efficiently (8–16 parallel games)
- The value baseline reduces variance from the noisy multi-agent reward signal

The reward shaping seen in the diagram above is critical. The raw sparse reward (survived = win) gives almost no gradient signal for 490 steps. Dense shaping bridges that gap. Then at step ~400 in training you add a **tiebreak shaping term** that mirrors the actual arena rule: `kills > boxes_destroyed > items_collected > bombs_placed`. This directly optimizes the metric that decides ladder ranking when multiple agents survive to step 500.

---

## Phase 4 — League play

Once you have a solid PPO policy, run a small population of 3–5 agents with different hyperparameter seeds (different entropy coefficients → different risk tolerance). They play against each other and the historical snapshot pool. This prevents your single agent from exploiting a fixed strategy that higher-rated opponents at the BTC arena would punish.

---

## What to clone from your rule agents

Your existing agents have logic worth preserving as **auxiliary losses or reward shaping**, not just BC targets:

| Rule logic | RL equivalent |
|---|---|
| `_can_escape_after_bomb` | Penalty reward if agent dies within 7 steps of placing a bomb |
| `_danger_schedule` (time-expanded) | Pre-computed danger channels fed as input (channels 8–14) |
| `_trap_bonus` | +0.4 reward bonus when an enemy is in your blast with ≤1 exit |
| Tiebreak `kills > boxes > items > bombs` | Shaped reward at step > 400 |
| `_item_targets` priority (capacity when low) | Item-type-aware reward: capacity pickup worth more when bombs_left ≤ 1 |

---

## Practical training timeline

Given the June 21 deadline and the 100ms/step inference constraint, the most realistic path is:

1. **Days 1–2:** BC on rule-agent rollouts. Fast to train, gives a strong floor.
2. **Days 3–7:** DQN self-play on Colab/Kaggle GPU with the BC checkpoint.
3. **Days 8–14:** Switch to PPO, add shaped tiebreak rewards, run parallel envs.
4. **Days 14–21:** League play refinement, submit snapshots as you improve.

The 100ms inference limit means your network must stay small — the CNN+LSTM above fits comfortably within ~2ms on CPU, leaving plenty of headroom. Avoid transformers or anything requiring expensive attention computation per step.

Click any box in the diagram above to drill deeper into any component.