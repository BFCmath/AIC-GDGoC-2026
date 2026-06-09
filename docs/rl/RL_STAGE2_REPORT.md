# RL Stage-2 Patch Report

## Implemented

- Added stage-wise training driver: `scripts/participant/stage_train_bc_ppo.py`.
- Added tuned curricula:
  - `configs/curriculum_stagewise_v2.json` for Kaggle/GPU.\n  - `configs/curriculum_sandbox_stagewise.json` for CPU smoke testing.
- Added fixed-stage PPO training via `--fixed-stage`.
- Added per-stage and best checkpoint output.
- Fixed D4 anti-diagonal augmentation mismatch.
- Adjusted BC to retain safe trap/pressure bombs.
- Added mild class-balanced BC loss.
- Added BTC tie-break aligned terminal/stat reward shaping.
- Made export more conservative and CPU-safe.

## Sandbox training actually run

Due to CPU-only constraints, I ran a compact smoke version:

```bash
python scripts/participant/stage_train_bc_ppo.py \
  --profile smoke --device cpu --end-stage 1 \
  --eval-matches 2 --torch-threads 1 \
  --export-dir exports/stagewise_smoke_agent
```

Then continued one tiny fixed stage:

```bash
python -u scripts/participant/train_bc_ppo.py \
  --mode ppo --device cpu --seed 2026 --torch-threads 1 \
  --checkpoint checkpoints/stagewise/stage1.pt \
  --save-checkpoint checkpoints/stagewise/stage2_tiny.pt \
  --curriculum-config configs/curriculum_sandbox_stagewise.json \
  --fixed-stage 2 --ppo-updates 4 --ppo-envs-per-update 3 \
  --ppo-horizon 24 --ppo-epochs 2 --ppo-minibatch-size 256 \
  --ppo-lr 0.0002 --ent-coef 0.016 \
  --eval-interval 2 --eval-matches 3
```

## Result

The tiny CPU-trained model is valid and sometimes wins, but it is not a reliable replacement for the rule/search codex agents yet. This is expected from the very small CPU budget. The useful deliverable here is the corrected stage-wise training infrastructure and safer export path for a real Kaggle GPU run.

