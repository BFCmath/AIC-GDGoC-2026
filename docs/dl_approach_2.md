## Best strategy found

The strongest practical strategy is:

**Safety-shielded neural policy trained by behavior cloning from `4.py / StatKillerHybridV4`, then fine-tuned with population self-play PPO.**

Do **not** train pure DQN from scratch as the main submission. In this game, the objective is an online ladder with dynamic opponents, 100ms/step limit, 500-step matches, and timeout tie-breaks by **kills → boxes → items → bombs**. So the winning policy must optimize survival plus material/stat advantage, not just “kill enemies.” 

The game itself also strongly rewards exact tactical reasoning: bomb timers, chain reactions, safe escape after bombing, item collection, and tiebreak farming all matter. 

## Local validation I ran

I tested the uploaded agents against strong local baselines.

| Agent                         |                           Quick arena result |
| ----------------------------- | -------------------------------------------: |
| `4.py / StatKillerHybridV4`   |                          best overall signal |
| `1.py / TimeSafeHybrid`       |             strong survival, less aggressive |
| `2.py / TacticalTieBreakerV2` | good item/farming logic, weaker danger model |
| built-in strong rule agents   |                       generally below `4.py` |

Important results:

| Test                                   |                             Result |
| -------------------------------------- | ---------------------------------: |
| `4.py` vs strong mix, 12 matches       |   avg rank `0.167`, win rate `83%` |
| `4.py` seat-balanced, 32 matches       |   avg rank `0.312`, win rate `72%` |
| `4.py` vs `1.py` direct shared matches | `4.py` beat `1.py` in rank `19/24` |

Timing was also safe:

| Agent  | Avg ms/step | Max spike |
| ------ | ----------: | --------: |
| `4.py` |    `0.82ms` |  `3.46ms` |
| `1.py` |    `0.69ms` |  `3.03ms` |
| `2.py` |    `0.15ms` |  `0.53ms` |

So `4.py` is the best current teacher. It already implements the right backbone: time-expanded danger schedule, exact escape BFS, immediate item pickup, safe bombing only if it creates box/kill/trap value, target BFS, and enemy bomb threat avoidance. 

## Why pure DQN/PPO from scratch is likely weaker

Pure RL from scratch will waste most samples learning things that `4.py` already solves: not walking into bombs, escaping after placing bombs, not bombing in dead ends, farming items, and understanding chain reactions. In a 4-agent non-stationary game, naïve DQN also has a moving-target problem: the value of an action changes depending on all other agents’ policies.

The best learning-based approach is therefore **not** “replace the search agent.” It is:

> use search/rules as a safety and expert prior, then train a neural policy to learn opponent-specific aggression, timing, and high-level mode switching.

## Final training recipe

### 1. Generate expert data

Use `4.py` as the main expert and add diversity from `1.py`, `2.py`, and all built-in baselines.

Generate matches like:

```text
[4.py, TacticalRule, GeniusRule, SmarterRule]
[4.py, 1.py, 2.py, TacticalRule]
[4.py, previous_checkpoint, random_baseline, top_baseline]
[noisy_4.py, noisy_4.py, 1.py, 2.py]
```

Save every state as:

```text
obs, legal_action_mask, expert_action, rank_outcome, final_stats
```

Use board symmetries aggressively: rotate/reflect the 13×13 map and remap actions. This gives 4–8× more data almost for free.

### 2. Train behavior cloning first

Model:

```text
Input channels:
- grass / wall / box / item_radius / item_capacity
- my position
- each enemy position, or enemy union + nearest enemy map
- bomb timer channels t=1..7
- bomb owner/self/enemy
- predicted blast danger for next 1..8 steps
- legal action mask
- safe-after-action mask
- box value map
- item value map
- enemy trap opportunity map

Network:
small ResNet/IMPALA CNN over 13×13
+ scalar MLP for my bombs_left, radius, turn, alive counts, stats
+ policy head over 6 actions
+ value head for expected final rank
```

Loss:

```text
L = CE(expert_action)
  + 0.5 * MSE(value, normalized_final_rank)
  + illegal_action_penalty
```

The key is to train the network to imitate `4.py` until it reaches near-zero illegal/death-prone actions.

### 3. Add a hard safety shield

At inference, the neural model should never be allowed to choose obviously suicidal actions.

For each candidate action:

```text
reject if illegal
reject if position is in danger at t+1
reject PLACE_BOMB if no escape path exists after bomb
prefer action with escape path over action without escape path
```

Then choose among the remaining actions using the neural policy.

This is the strongest hybrid pattern:

```python
action = neural_policy(obs, mask=safe_actions)
if action unsafe or low confidence:
    action = StatKillerHybridV4.act(obs)
return action
```

This keeps neural exploration from destroying the main advantage of the current best agent.

### 4. Fine-tune with population self-play PPO

Use PPO, not DQN, for the main fine-tuning stage.

Opponent population:

```text
30% StatKillerHybridV4 variants
20% TimeSafeHybrid / TacticalTieBreaker
20% built-in strong baselines
20% previous PPO checkpoints
10% random/noisy/weird agents
```

This mirrors the real ladder idea: rating-near, top, and random opponents. 

Reward:

```text
terminal:
  unique rank 0 win: +2.0
  shared rank 0 draw: +0.7
  rank 1: +0.2
  rank 2: -0.6
  rank 3: -1.2
  death by own bomb: extra -0.8

dense shaping:
  + kill
  + boxes destroyed, but only if escape remains possible
  + item collected
  + capacity item slightly > radius item early
  + safe escape from danger
  - standing in future blast
  - placing bomb without escape
  - chasing enemy while behind in safety/material
```

Important: include final stat advantage in reward because timeout ranking uses kills, boxes, items, bombs. 

### 5. Train modes, not just actions

The neural part should learn a latent “mode”:

```text
EARLY_FARM:
  break boxes, collect capacity/radius, avoid fights

MID_CONTROL:
  take center/space, deny items, bomb high-value boxes

TRAP_ATTACK:
  attack when enemy has low mobility or is near corridor/dead-end

LATE_TIEBREAK:
  if ahead: survive and farm safe bombs
  if behind: force kill/trap, accept more risk
```

This is where learning can beat pure heuristic code: dynamic risk control.

## Outside-the-box exploit

The ladder is not only about killing. It is about **not being the first to die**, then winning timeout tiebreaks.

So train the agent to identify when it is already ahead. If it has better `(kills, boxes, items, bombs)` than likely survivors, it should **de-escalate** and avoid direct duels. If it is behind, it should intentionally create volatile bomb-chain situations near enemies.

This “score-aware risk switching” is the biggest improvement I would target over `4.py`.

## Submission recommendation

For the next submission:

1. Submit `4.py` or a lightly improved version as the safe baseline.
2. In parallel, train **BC → PPO self-play** using `4.py` as teacher.
3. Final deployed agent should be:

```text
Neural policy
+ legal/safety action mask
+ exact bomb escape checker
+ fallback to StatKillerHybridV4
```

Do **not** submit a raw neural model without the shield. The strongest competition agent is likely a **neural high-level policy wrapped around exact tactical search**, not a pure end-to-end DQN.
