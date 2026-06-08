Here is the DL plan I would use for this competition.

## Recommendation: hybrid RL, not pure end-to-end RL

Use a **learned policy only for tactical choice**, wrapped by a **rule-based safety layer**. Bomberland has sparse rewards, lethal bombs, 100 ms CPU inference, and multi-agent instability, so a pure DQN/PPO agent that directly learns everything from scratch will likely underperform strong BFS/rule agents.

The competition setup is a 4-agent, 13×13 Bomberman-style environment with full state observation, 6 discrete actions, bombs, items, box farming, survival, and elimination-based ranking. The agent must also act fast, with about 100 ms per step. 

Your submitted agent should look like this:

```text
obs
 ├─ rule safety module
 │   ├─ legal action mask
 │   ├─ bomb danger map for future 1..7 steps
 │   ├─ escape-path checker
 │   └─ bomb-placement veto
 │
 ├─ neural policy/value model
 │   ├─ CNN over 13×13 feature planes
 │   ├─ MLP over scalar features
 │   └─ outputs 6 action logits + value
 │
 └─ final action selector
     ├─ forced escape if in danger
     ├─ masked neural action otherwise
     └─ fallback to Tactical/Genius-style rule action
```

## Important repo finding

Do **not** use the provided `agent/dqn_agent` unchanged as your final approach.

The current DQN reference is useful as a starter, but it has major limitations:

1. Its encoder effectively models only one opponent, not all three opponents.
2. The training loop appears simplified around one controlled agent and one explicit enemy, while the environment still has 4 players.
3. It has no strong action mask or bomb-safety veto.
4. It uses a basic DQN objective, which is fragile in this adversarial multi-agent setting.

So treat it as a template for loading PyTorch models, not as the main solution.

## Model architecture

Use a small **CNN + MLP actor-critic** model.

### Spatial input channels

For each observation, build a tensor shaped roughly:

```text
(C, 13, 13)
```

Suggested channels:

```text
Terrain:
1. grass
2. wall
3. box
4. radius item
5. capacity item

Agents:
6. my position
7. opponent 0 position
8. opponent 1 position
9. opponent 2 position

Bombs:
10. bomb exists
11. bomb timer / 7
12. bomb radius / 5
13. my bomb
14. enemy bomb

Danger:
15. blast at t=1
16. blast at t=2
17. blast at t=3
18. blast at t=4
19. blast at t=5
20. blast at t=6
21. blast at t=7

Planning helpers:
22. reachable in ≤1 step
23. reachable in ≤2 steps
24. reachable in ≤3 steps
25. safe reachable cells
26. cells where placing bomb hits box
27. cells where placing bomb threatens enemy
```

You do not need all channels on day one. Start with terrain, agents, bombs, and danger maps. Add planning-helper channels after the first working model.

### Scalar features

Use an auxiliary vector:

```text
[
  agent_id one-hot,
  normalized step count,
  my bombs_left,
  inferred max bomb capacity,
  my radius,
  number of alive opponents,
  nearest enemy distance,
  nearest item distance,
  nearest box-bomb-spot distance,
  current danger timer,
  can_escape_if_place_bomb,
  boxes_hit_if_place_bomb,
  enemies_hit_if_place_bomb
]
```

The observation does not directly expose current step, so maintain `self.step_count` inside the agent and reset it heuristically when the map/players return to starting positions or when the agent object is recreated.

### Network

Keep it small for CPU inference:

```python
Conv2d(C, 32, 3, padding=1) + ReLU
Conv2d(32, 64, 3, padding=1) + ReLU
Conv2d(64, 64, 3, padding=1) + ReLU
Flatten

Scalar MLP:
Linear(aux_dim, 64) + ReLU
Linear(64, 64) + ReLU

Combined:
Linear(flattened_cnn + 64, 256) + ReLU

Policy head:
Linear(256, 6)

Value head:
Linear(256, 1)
```

Use `torch.inference_mode()` and `model.eval()` in submission.

## Algorithm choice

I would choose **PPO with action masking**, not vanilla DQN.

Reasons:

* The action space is tiny, so policy-gradient exploration is manageable.
* PPO handles shaped rewards and non-stationary opponents better than basic DQN.
* You can train against a mixture of baselines and self-play snapshots.
* The value head gives better training signal in long 500-step episodes.

If you want to stay closer to the starter DQN, upgrade it to **Double DQN + Dueling DQN + n-step returns + action masking**, but PPO is the cleaner plan.

## Safety layer

This is the most important part.

Before using the neural model, compute:

1. **Valid action mask**
   Mask actions that move into walls, boxes, existing bombs, or out of bounds.

2. **Danger map**
   For each bomb, compute blast tiles using its radius and timer. Include chain-reaction approximations if possible.

3. **Escape override**
   If the current tile is dangerous, ignore the neural policy and use BFS to move toward the nearest safe reachable cell.

4. **Bomb-placement veto**
   Only allow `PLACE_BOMB` if:

   * `bombs_left > 0`
   * no existing bomb is on the current tile
   * the bomb can hit at least one box or threaten an enemy
   * there is a safe escape route before the bomb explodes

5. **Final fallback**
   If the neural action is invalid or unsafe, use a Tactical/Genius-style rule action.

This alone will outperform many naive RL agents.

## Training pipeline

### Phase 1: behavior cloning warm start

Generate matches using the best rule agents:

* `TacticalRuleAgent`
* `GeniusRuleAgent`
* `SmarterRuleAgent`
* your own improved rule agent

Collect `(obs, expert_action)` pairs.

Train the neural policy with cross-entropy:

```text
loss = CE(masked_policy_logits, expert_action)
```

Goal: make the model imitate a competent rule agent before RL. This avoids early self-destruction and random bombing.

Target dataset size:

```text
50k–300k steps
```

### Phase 2: PPO fine-tuning

Train your agent against a randomized opponent pool:

```text
40% strong baselines: Tactical, Genius, Smarter
30% random baseline mixture
20% previous snapshots of your own agent
10% current self-play clone
```

Randomize your controlled `agent_id` from 0 to 3. This matters because starting corners differ.

Use curriculum:

```text
Stage 1: 150-step games, simple/smarter opponents
Stage 2: 300-step games, tactical/genius opponents
Stage 3: 500-step games, baseline mixture + self-play snapshots
```

### Phase 3: snapshot league

Every N training updates, save a model snapshot. During training, sample old snapshots as opponents.

This prevents the policy from overfitting to one baseline style.

## Reward shaping

Use terminal reward aligned with ranking:

```text
unique 1st place:      +10
shared best draw:      +4
2nd place:             +2
3rd place:             -2
4th place / first die: -6
self-death:            -8
enemy killed:          +4
```

Add small dense rewards:

```text
+0.05  survive one step, but cap total survival reward
+0.20  collect item
+0.10  destroy box
+0.30  place bomb that will hit box and has escape path
+1.00  place bomb that traps/threatens enemy and has escape path
+0.20  leave danger zone
-0.50  enter danger zone
-1.00  place bomb with no escape
-0.02  useless STOP when not in danger
```

Be careful: do **not** reward bomb placement by itself. Reward only useful bomb placement.

## Action mapping warning

The code/guide has a coordinate bug. In practice, use action IDs, not action names:

```text
0 = STOP
1 = moves visually UP
2 = moves visually DOWN
3 = moves visually LEFT
4 = moves visually RIGHT
5 = PLACE_BOMB
```

This is critical. If your training code and submission code disagree on this, the model will look “trained” but play terribly.

## Evaluation protocol

Use these local checks repeatedly:

```bash
python -m scripts.participant.estimate_rankings \
  --agent_path path/to/your_agent \
  --num_matches 200
```

```bash
python -m scripts.participant.estimate_agent_time \
  path/to/your_agent \
  --opponents None None None \
  --num_matches 10
```

Also run targeted matchups:

```bash
python -m scripts.participant.run_local_match \
  --agent_paths path/to/your_agent TacticalRuleAgent GeniusRuleAgent SmarterRuleAgent \
  --num_episodes 100 \
  --visualize false
```

Track:

```text
win rate
average rank
self-death rate
timeout count
bombs placed
boxes destroyed
items collected
kills
survival steps
```

The most important metric early is **self-death rate**. If it is high, improve the safety layer before training more.

## Practical milestone plan

### Milestone 1: strong hybrid baseline

Build a rule-based agent based on Tactical/Genius logic:

* danger map
* BFS escape
* item targeting
* bomb only with escape
* box farming
* enemy pressure

Submit this first. It gives you a reliable benchmark.

### Milestone 2: neural imitation agent

Train the CNN policy to imitate your hybrid rule agent. At submission time:

```text
if danger: BFS escape
else: masked neural action
```

This should match or slightly exceed your rule baseline if the neural model learns better tie-breaking.

### Milestone 3: PPO fine-tuning

Fine-tune from the behavior-cloned model using PPO against baseline mixtures and snapshots.

### Milestone 4: competition hardening

Before final submission:

* freeze model
* remove training code
* keep only `agent.py` and model weights
* force CPU inference
* verify no file writing
* verify no network calls
* benchmark under 100 ms
* submit early enough to accumulate matches

## Final strategy

Your best chance is not “deep learning replaces planning.” It is:

```text
planning guarantees survival;
deep learning chooses among safe tactical options.
```

In Bomberland, unsafe cleverness loses immediately. A small masked PPO policy, initialized by behavior cloning and protected by BFS safety logic, is the most practical DL approach for this competition.
    