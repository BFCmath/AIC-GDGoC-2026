# Stage-wise RL training plan for BomberGame

This repo now supports a more practical staged BC + PPO workflow than the single `--mode full` run.

## Why this version is different

Key fixes/tuning:

1. **Correct D4 augmentation**
   - Fixed the anti-diagonal spatial transform so its action labels are no longer inconsistent.

2. **Better BC data**
   - BC no longer throws away every safe trap/pressure bomb that does not immediately hit a box/enemy.
   - BC uses mild inverse-frequency action weighting so STOP/move actions do not dominate bomb examples.

3. **Stage-wise PPO**
   - Added `--fixed-stage`, `--init-stage`, `--eval-matches`, `--stage-checkpoint-dir`, and `--best-checkpoint`.
   - Each stage can be trained independently from the previous checkpoint.
   - Snapshots and eval checkpoints are saved for rollback/selection.

4. **Better curriculum**
   - `configs/curriculum_stagewise_v2.json` replaces the too-short 5/7/8/10-step curriculum.
   - Stages move from short-rule-agent games toward longer codex-heavy games.

5. **Safer export**
   - Exported `agent.py` uses a conservative neural override gate.
   - It falls back to a teacher league with `8.py`, `7.py`, and `4.py`.
   - CPU threads are pinned to avoid 100ms/server spikes.

## Recommended Kaggle medium run

```bash
%cd AIC-GDGoC-2026
!pip install -r requirements.txt

!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile kaggle_medium \
  --device cuda \
  --eval-matches 12 \
  --export-dir exports/stagewise_hybrid_agent
```

This runs:

- BC: 50 matches, 28 epochs, max_steps 220
- PPO stage 0: 30 updates
- PPO stage 1: 40 updates
- PPO stage 2: 50 updates
- PPO stage 3: 60 updates
- PPO stage 4: 70 updates
- PPO stage 5: 80 updates
- PPO stage 6: 90 updates
- PPO stage 7: 120 updates

## Longer run

```bash
!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile kaggle_long \
  --device cuda \
  --eval-matches 16 \
  --export-dir exports/stagewise_hybrid_agent_long
```

## Resume from a stage

```bash
!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile kaggle_medium \
  --device cuda \
  --skip-bc \
  --start-stage 4 \
  --end-stage 7 \
  --eval-matches 12 \
  --export-dir exports/stagewise_hybrid_agent_resume
```

## Benchmark after export

```bash
!python -m scripts.participant.benchmark \
  --agents exports/stagewise_hybrid_agent agent/codex/4.py agent/codex/7.py agent/codex/8.py \
  --matches 50 \
  --max_steps 400 \
  --timeout \
  --timeout-ms 100
```

## Submission zip

```bash
cd exports/stagewise_hybrid_agent
zip -r ../../stagewise_hybrid_agent_submission.zip agent.py model.pt 4.py 7.py 8.py
```

## Notes

- Do not trust a checkpoint only because training reward improved. Select using benchmark + timeout.
- If the neural-only/masked eval is still weak, keep the conservative export gate. It will behave like a strong teacher-league agent with occasional learned overrides.
- If `fallback_rate` remains high through late stages, increase BC matches/epochs before increasing PPO.
