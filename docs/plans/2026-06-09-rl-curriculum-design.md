# RL Curriculum & Anti-Cowardice Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Overhaul the RL pipeline in `train_bc_ppo.py` to eliminate reward hacking (wiggling/cowardice), implement an objective 8-stage curriculum based on win rates, and bootstrap from Codex 7, Codex 4, and Codex 8.

**Architecture:** 
1. Replace survival (+0.03) with step penalty (-0.01) to encourage speed.
2. Introduce a position history tracker that penalizes entering recently visited tiles to stop repetitive moving.
3. Introduce dynamic reward shaping that depends on the current **Stage**:
   - Stages 0-5: Heavy reward for 1st place, heavy penalty for 2nd place and below. In boss stages, killing the boss agent yields high rewards.
   - Stage 6: High reward for 1st place, moderate reward for 2nd place.
   - Stage 7: Scaled rewards aligned with rank, heavy penalty for last place.
4. Introduce a curriculum evaluation phase that runs 20 deterministic matches against specific baselines every $N$ updates. Stage advances only if Win Rate >= 40% (or defined threshold).
5. Update BC pre-training to sample 45% Codex 7, 30% Codex 4, 25% Codex 8.

**Tech Stack:** PyTorch, Python (Bomberland Env)

---

### Task 1: Update Reward Shaping to Dynamic Anti-Cowardice Mode

**Files:**
- Modify: `scripts/participant/train_bc_ppo.py`

**Step 1: Write the minimal implementation for reward shaping**
Modify `shaped_reward` function to take `stage` into account:
- Add a Wiggle Tracker penalty: If `action != 0` and `cur_pos` is in `prev_visited_positions`, `reward -= 0.05`
- Stages 0-5: Rank 0 gets `+15.0`. Rank > 0 gets `-10.0`. If Stage is 2, 3, 4, or 5 and the agent kills the specific boss (Codex 4, 8, 7, 7), they get an extra `+10.0` kill bounty.
- Stage 6: Rank 0 gets `+15.0`. Rank 1 gets `+5.0`. Rank > 1 gets `-10.0`.
- Stage 7: Rank 0 gets `+15.0`. Rank 1 gets `+5.0`. Rank 2 gets `-5.0`. Rank 3 gets `-15.0`.
- Change survival reward to `-0.01` (to encourage speed)

**Step 2: Commit**
```bash
git commit -am "feat: update reward shaping to dynamic anti-cowardice mode"
```

### Task 2: Implement WiggleTracker in the Environment Runner

**Files:**
- Modify: `scripts/participant/train_bc_ppo.py`

**Step 1: Implement WiggleTracker logic**
In the environment step loop inside PPO runner:
- Maintain a `deque(maxlen=4)` of `prev_visited_positions` for each agent.
- Pass `prev_visited_positions` into `shaped_reward`.

**Step 2: Commit**
```bash
git commit -am "feat: add WiggleTracker to prevent repetitive moving"
```

### Task 3: Implement Objective Win-Rate Evaluator & 8-Stage Curriculum

**Files:**
- Modify: `scripts/participant/train_bc_ppo.py`

**Step 1: Implement Evaluator Class**
Create a function `evaluate_agent_win_rate(model, opponents, num_matches=20)` that runs deterministic matches without PPO updates, measuring win rate.

**Step 2: Implement Stage-by-Stage Curriculum Logic**
Define the 8 stages:
- **Stage 0**: `["RandomAgent", "SimpleRuleAgent"]`
- **Stage 1**: `["TacticalRuleAgent", "SmarterRuleAgent", "BoxFarmerAgent"]`
- **Stage 2**: `["agent/codex/4.py", "SimpleRuleAgent", "SimpleRuleAgent"]`
- **Stage 3**: `["agent/codex/8.py", "SimpleRuleAgent", "SimpleRuleAgent"]`
- **Stage 4**: `["agent/codex/7.py", "SimpleRuleAgent", "SimpleRuleAgent"]`
- **Stage 5**: `["agent/codex/7.py", "TacticalRuleAgent", "SmarterRuleAgent"]`
- **Stage 6**: `["agent/codex/8.py", "agent/codex/4.py", "TacticalRuleAgent"]`
- **Stage 7**: `["agent/codex/7.py", "agent/codex/4.py", "agent/codex/8.py"]`

Update main training loop to spawn opponents based on current stage, and run evaluation every `EVAL_INTERVAL` updates to advance to the next stage.

**Step 3: Commit**
```bash
git commit -am "feat: implement objective win-rate curriculum evaluator and 8 stages"
```

### Task 4: Update BC Initialization

**Files:**
- Modify: `scripts/participant/train_bc_ppo.py`

**Step 1: Update BC specs**
In `collect_expert_dataset`, change sampling:
- 45% `agent/codex/7.py`
- 30% `agent/codex/4.py`
- 25% `agent/codex/8.py`

**Step 2: Commit**
```bash
git commit -am "feat: focus BC pre-training on top-tier codex agents"
```
