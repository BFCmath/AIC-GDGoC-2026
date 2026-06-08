After checking the rules, benchmark, and repo, my conclusion is:

**Do not replace the rule/search agent with pure end-to-end RL.**
The best path is a **hybrid RL agent**: keep deterministic safety/search, then train a neural policy to choose among safe tactical actions. In Bomberman/Pommerman-style games, pure model-free RL often dies by exploration and learns bad local optima; successful learning agents usually use **imitation learning + PPO + reward shaping + action filtering / pruning + curriculum / self-play**. ([arXiv][1])

## 1. Benchmark implication

Your rank 317 result is not only because the algo is weak; it also has only **17 games**, so sigma is still high. But the important signal is:

```text
Win rate: 23.5%
Avg rank: 0.8824
Avg steps: 408.6
```

The benchmark is an **online arena**, not a fixed test set. Opponents are sampled as **40% near rating, 30% top agents, 30% random**, and the rating uses TrueSkill under a 100ms/step CPU limit. So the agent must generalize against strong unseen styles, not just beat local baselines. 

The game itself is small enough for strong search features: **4 agents, 13×13 grid, 6 actions, 500 max steps, bomb timer/radius/capacity, item farming, and strict death/tie-break rules**. That means safety + planning features are extremely valuable and should not be discarded.   

## 2. Repo finding

I inspected the uploaded repo. Your submitted `agent/codex/4.py` is already a serious rule/search hybrid with danger scheduling, escape BFS, box farming, item targeting, bomb escape checks, and trap scoring.

I also ran a quick local sanity benchmark against repo baselines. Small sample, not leaderboard-equivalent:

```text
vs Tactical + Genius + Smarter: rank-0 rate 65%, avg rank 0.35
vs 3× Genius:                  rank-0 rate 90%, avg rank 0.15
vs 3× Tactical:                rank-0 rate 75%, avg rank 0.30
vs 3× Random:                  rank-0 rate 100%
mean action time:              ~0.2–0.35 ms
```

So the current agent is **not bad locally**. The leaderboard gap likely comes from **out-of-distribution top agents**, not from failing basic rules.

Also, the repo’s DQN starter should not be used unchanged. It is too weak for this competition: it encodes essentially one opponent, has weak/no safety masking, and uses vanilla DQN in a sparse, delayed, adversarial multi-agent setting. That is exactly the failure mode Pommerman papers warn about: sparse/deceptive rewards, delayed bomb effects, and frequent suicide during exploration. ([arXiv][1])

## 3. Best RL approach

Build this:

```text
obs
 ├─ deterministic feature engine
 │   ├─ legal action mask
 │   ├─ danger map t=1..7
 │   ├─ escape path checker
 │   ├─ bomb usefulness checker
 │   └─ candidate action features
 │
 ├─ neural policy/value model
 │   ├─ tiny CNN over 13×13 planes
 │   ├─ scalar feature MLP
 │   └─ masked PPO logits over 6 actions
 │
 └─ safety wrapper
     ├─ forced escape if currently threatened
     ├─ forbid suicidal bomb
     ├─ forbid invalid movement
     └─ fallback to 4.py rule action
```

This matches what worked in related Pommerman research: imitation warm-start followed by PPO, with heuristic action filters, reward shaping, and curriculum learning. One Pommerman paper reports that this combination beat heuristic and pure RL baselines using 100,000 training games. ([arXiv][2])

A later Pommerman training paper also emphasizes **curriculum learning + population-based self-play + matchmaking**, because sparse rewards and opponent self-suicide can make naive RL misleading. ([arXiv][3])

## 4. The most elegant variant: RL action reranker

Instead of asking RL to learn Bomberman from raw grid only, make it learn **which safe plan is best**.

For each of the 6 actions, compute features like:

```text
is_legal
is_safe_next_step
min_escape_time
safe_space_after_action
boxes_hit_if_bomb
enemy_in_blast_if_bomb
trap_score_if_bomb
distance_to_nearest_item_after_action
distance_to_nearest_enemy_after_action
future_danger_margin
tie_break_value: kills/boxes/items/bombs potential
```

Then the neural model scores each action:

```text
score(s, a) = MLP([global_state_embedding, action_features])
```

Apply mask:

```text
invalid actions -> -inf
suicidal bomb -> -inf
unsafe movement -> -inf unless no alternative
```

Then sample/argmax from the masked distribution.

This is better than a generic CNN-only PPO because the model does not waste capacity rediscovering basic Bomberman physics. Invalid action masking is also a standard and theoretically justified method for large/structured discrete action spaces. ([arXiv][4])

## 5. Training recipe

### Phase A — Behavior cloning from strong teachers

Use your `4.py` plus `GeniusRuleAgent`, `TacticalRuleAgent`, `SmarterRuleAgent`, and maybe several mutated versions of `4.py`.

Collect:

```text
(obs, action, legal_mask, safety_features, teacher_action)
```

Target size:

```text
200k–1M steps
```

Train with:

```text
loss = cross_entropy(masked_logits, teacher_action)
```

This gives the policy a non-suicidal prior.

### Phase B — DAgger-style teacher correction

Let the neural policy play, but when it reaches unfamiliar states, relabel those states using your best rule/search teacher.

This is important because pure BC only learns states visited by the teacher. DAgger fixes distribution shift.

### Phase C — PPO fine-tuning

Use PPO, not vanilla DQN.

Opponent sampling should mimic the real benchmark:

```text
30% top local agents: 4.py, Genius, Tactical, Smarter
25% self-play snapshots
20% noisy/aggressive variants
15% random/weak agents
10% current policy clones
```

Randomize your controlled `agent_id` from 0 to 3 every episode. Otherwise the model overfits to one corner.

Curriculum:

```text
Stage 1: 150-step games, weak/noisy opponents, low bomb aggression
Stage 2: 300-step games, Genius/Tactical/4.py opponents
Stage 3: 500-step games, snapshot league + top opponents
Stage 4: hard mode, only top snapshots and adversarial bombers
```

### Phase D — Population / league training

Save snapshots every N PPO updates. Keep an Elo/TrueSkill table locally. Train mostly against agents near or above your rating, plus a small random pool. This directly matches the leaderboard’s near/top/random sampling logic. 

## 6. Reward design

Use ranking-aligned reward first:

```text
unique rank 0 / sole winner: +10
draw best rank:              +3
rank 1:                      +1
rank 2:                      -2
rank 3 / first dead:          -6
self-death:                  -8
kill opponent:               +4
```

Dense shaping, small magnitude:

```text
+0.03 survive one step, capped
+0.15 collect item
+0.08 destroy box
+0.25 place bomb that hits box and has escape path
+0.80 place bomb that traps/threatens enemy and has escape path
+0.30 leave danger
-0.70 enter unavoidable danger
-1.50 place bomb with no escape
-0.02 useless STOP when safe
```

Do **not** reward bomb count blindly. Bomb count is only a tie-breaker after survival/kills/boxes/items, so over-rewarding bombs creates suicidal spam.

## 7. Important implementation warnings

The engine has an action-coordinate mismatch. In actual engine behavior, use action IDs like this:

```text
0 = STOP
1 = move row -1  / visually UP
2 = move row +1  / visually DOWN
3 = move col -1  / visually LEFT
4 = move col +1  / visually RIGHT
5 = PLACE_BOMB
```

Train and submit with the same mapping. A mismatch here will destroy an otherwise good model.

Also, submission inference must stay tiny:

```python
torch.set_num_threads(1)
model.eval()
with torch.inference_mode():
    ...
```

A tiny CNN/MLP or action-reranker will be safely below 100ms. Your current rule agent is already under 1ms locally, so the hybrid can stay fast.

## 8. Priority experiments

Run these in order:

1. **Fix evaluator first**: local eval must use official tie-break ordering: kills → boxes → items → bombs, not just survival.
2. **BC-only neural reranker**: imitate `4.py`; verify it does not self-destruct.
3. **BC + safety wrapper**: submit candidate only after self-death rate is near zero.
4. **PPO fine-tune vs baseline pool**: compare against frozen BC.
5. **Snapshot league**: train against older versions and adversarial rule variants.
6. **Ablate safety**: prove safety wrapper improves win rate and not just average survival.
7. **Submit early**: leaderboard TrueSkill needs more than 17 games to stabilize.

My recommended final submission architecture:

```text
4.py safety/search core
+ learned PPO action reranker
+ hard action mask
+ bomb veto
+ fallback to rule action
```

This is the highest-probability route. Pure RL is elegant on paper, but in this benchmark it will likely lose to well-engineered hybrids because bombs create delayed catastrophic exploration. The learning component should optimize tactical choice, not relearn survival physics from scratch.

[1]: https://arxiv.org/pdf/1904.05759 "Safer Deep RL with Shallow MCTS: A Case Study in Pommerman"
[2]: https://arxiv.org/abs/1911.04947 "[1911.04947] Accelerating Training in Pommerman with Imitation and Reinforcement Learning"
[3]: https://arxiv.org/abs/2407.00662 "[2407.00662] Multi-Agent Training for Pommerman: Curriculum Learning and Population-based Self-Play Approach"
[4]: https://arxiv.org/abs/2006.14171 "[2006.14171] A Closer Look at Invalid Action Masking in Policy Gradient Algorithms"
