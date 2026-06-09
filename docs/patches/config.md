Use **stage-wise training**, not one giant run. The best practical config is:

```python
BC_MATCHES = 80
BC_EPOCHS = 40
PPO_UPDATES = 900
PPO_ENVS = 16
PPO_HORIZON = 128
EVAL_MATCHES = 12
```

This is the config I would run first on Kaggle/Colab with GPU.

```bash
%cd AIC-GDGoC-2026
!pip install -r requirements.txt

!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile kaggle_medium \
  --device cuda \
  --eval-matches 12 \
  --torch-threads 1 \
  --export-dir exports/stagewise_hybrid_agent
```

Then benchmark:

```bash
!python -m scripts.participant.benchmark \
  --agents exports/stagewise_hybrid_agent agent/codex/4.py agent/codex/7.py agent/codex/8.py \
  --matches 50 \
  --max_steps 400 \
  --timeout \
  --timeout-ms 100
```

### My recommended configs

#### 1. Fast sanity run

Use this only to check that training works.

```bash
!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile smoke \
  --device cuda \
  --end-stage 2 \
  --eval-matches 4 \
  --torch-threads 1 \
  --export-dir exports/stagewise_smoke_agent
```

Expected: valid export, but probably not stronger than `4.py`.

---

#### 2. Main Kaggle run

This is the best balance.

```bash
!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile kaggle_medium \
  --device cuda \
  --eval-matches 12 \
  --torch-threads 1 \
  --export-dir exports/stagewise_hybrid_agent
```

Recommended target:

```text
BC matches: 80
BC epochs: 40
PPO updates per stage: ~80–160
PPO envs: 16
PPO horizon: 128
Stages: 0 → 7
```

This should produce checkpoints like:

```text
checkpoints/stagewise/stage0.pt
checkpoints/stagewise/stage1.pt
...
checkpoints/stagewise/stage7.pt
checkpoints/stagewise/best_stage*.pt
```

---

#### 3. Stronger overnight run

Use this after the medium run is stable.

```bash
!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile kaggle_strong \
  --device cuda \
  --eval-matches 20 \
  --torch-threads 1 \
  --export-dir exports/stagewise_hybrid_agent_strong
```

Recommended target:

```text
BC matches: 150–250
BC epochs: 50–80
PPO updates: 1500–2500 total
PPO envs: 16–32
PPO horizon: 128 or 192
Eval matches: 20
```

Do **not** push BC epochs too high with too few matches. Prefer:

```text
Good: 150 matches × 50 epochs
Risky: 10 matches × 250 epochs
```

The second one overfits badly to a small expert sample.

### Best curriculum direction

Your original curriculum was too short. For leaderboard-style play, use longer stages because official matches can reach 500 steps and tie-breaks use kills, boxes, items, then bombs. 

Better stage idea:

```json
{
  "0": {"max_steps": 80,  "win_rate": 0.65},
  "1": {"max_steps": 120, "win_rate": 0.60},
  "2": {"max_steps": 160, "win_rate": 0.55},
  "3": {"max_steps": 220, "win_rate": 0.50},
  "4": {"max_steps": 280, "win_rate": 0.45},
  "5": {"max_steps": 350, "win_rate": 0.40},
  "6": {"max_steps": 420, "win_rate": 0.35},
  "7": {"max_steps": 500, "win_rate": 0.35}
}
```

The important point: **do not force 0.8 win rate in early stages**. Against strong codex agents, that blocks progression and wastes GPU time.

### After training, select checkpoint manually

Do not assume the final stage is best. Benchmark several exports:

```bash
!python -m scripts.participant.benchmark \
  --agents exports/stagewise_hybrid_agent agent/codex/4.py agent/codex/7.py agent/codex/8.py \
  --matches 100 \
  --max_steps 500 \
  --timeout \
  --timeout-ms 100
```

Pick the model with best:

1. win count,
2. average rank,
3. no timeout,
4. good performance against both `4.py` and `7.py`, not only one.

### My strongest starting recommendation

Run this first:

```bash
!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile kaggle_medium \
  --device cuda \
  --eval-matches 12 \
  --torch-threads 1 \
  --export-dir exports/stagewise_hybrid_agent
```

Then, only if it beats or matches `7.py` but still loses to `4.py`, continue from the best checkpoint with more late-stage PPO:

```bash
!python -u scripts/participant/train_bc_ppo.py \
  --mode ppo \
  --device cuda \
  --checkpoint checkpoints/stagewise/best_stage7.pt \
  --save-checkpoint checkpoints/stagewise/late_stage_finetune.pt \
  --curriculum-config configs/curriculum_stagewise_v2.json \
  --fixed-stage 7 \
  --ppo-updates 600 \
  --ppo-envs-per-update 16 \
  --ppo-horizon 192 \
  --eval-interval 10 \
  --eval-matches 20 \
  --snapshot-interval 25 \
  --torch-threads 1 \
  --export-dir exports/stagewise_late_finetune_agent
```

That is the config most likely to improve ranking without overfitting too hard.
