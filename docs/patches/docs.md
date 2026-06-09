Pushed it further and packaged the stage-wise RL version.

I **did not claim victory falsely**: with the CPU sandbox budget, the tiny staged run is still **not consistently better than `4.py` / `7.py` / `8.py`**. But I made the training setup much more serious for Kaggle/GPU and fixed issues that were directly limiting BC/PPO quality.

### What I changed

Key improvements:

1. **Fixed BC augmentation bug**

   * The anti-diagonal D4 transform was inconsistent with its action remap.
   * This could inject wrong labels during BC.

2. **Improved BC data**

   * Safe trap/pressure bombs are no longer discarded just because they do not immediately hit a box/enemy.
   * Added mild inverse-frequency action weighting so BC does not under-learn bomb actions.

3. **Added stage-wise training**

   * New script: `scripts/participant/stage_train_bc_ppo.py`
   * Supports fixed-stage training, checkpointing per stage, best checkpoint saving, and resume.

4. **Replaced weak curriculum**

   * Added `configs/curriculum_stagewise_v2.json`
   * Progression is now meaningful: short simple games → longer codex-heavy games.
   * Your 5/7/8/10-step curriculum was too short to teach real 500-step ladder behavior.

5. **Improved PPO checkpointing**

   * Added:

     * `--fixed-stage`
     * `--init-stage`
     * `--eval-matches`
     * `--stage-checkpoint-dir`
     * `--best-checkpoint`

6. **Export is safer**

   * Conservative neural override gate.
   * Teacher fallback league uses `8.py`, `7.py`, `4.py`.
   * CPU threads pinned to reduce 100ms timeout risk.

This matters because the official-style benchmark rewards survival and tie-break stats: kills → boxes → items → bombs, under 100ms/step constraints. 

### Sandbox test result

I ran a compact staged CPU smoke training:

```bash
python scripts/participant/stage_train_bc_ppo.py \
  --profile smoke \
  --device cpu \
  --end-stage 1 \
  --eval-matches 2 \
  --torch-threads 1 \
  --export-dir exports/stagewise_smoke_agent
```

Then one tiny stage-2 continuation:

```bash
python -u scripts/participant/train_bc_ppo.py \
  --mode ppo \
  --device cpu \
  --checkpoint checkpoints/stagewise/stage1.pt \
  --save-checkpoint checkpoints/stagewise/stage2_tiny.pt \
  --curriculum-config configs/curriculum_sandbox_stagewise.json \
  --fixed-stage 2 \
  --ppo-updates 4 \
  --ppo-envs-per-update 3 \
  --ppo-horizon 24
```

Timeout benchmark passed:

```text
LeagueBCPPO_Assassin: 0 / 960 timeouts
StatKillerHybridV4:   0 / 874 timeouts
AntiTrapHybridV9:     0 / 960 timeouts
SeatAwareAssassinV20: 0 / 960 timeouts
```

In an 8-match timeout benchmark, the tiny trained export got:

```text
LeagueBCPPO_Assassin: 2 wins, avg rank 1.38
AntiTrapHybridV9:     2 wins, avg rank 1.50
SeatAwareAssassinV20: 2 wins, avg rank 1.25
StatKillerHybridV4:   2 wins, avg rank 1.88
```

That is **promising but not conclusive** because 8 matches is noisy and the model is barely trained.

### Recommended Kaggle run

Use this now:

```bash
%cd AIC-GDGoC-2026
!pip install -r requirements.txt

!python -u scripts/participant/stage_train_bc_ppo.py \
  --profile kaggle_medium \
  --device cuda \
  --eval-matches 12 \
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

### Files

* [Code-only patched repo](sandbox:/mnt/data/AIC-GDGoC-2026-RL-stagewise-code.zip)
* [Full patched repo with generated checkpoints/exports](sandbox:/mnt/data/AIC-GDGoC-2026-RL-stagewise.zip)
* [Submission zip from the tiny staged run](sandbox:/mnt/data/stagewise_hybrid_agent_submission.zip)
* [Patch diff](sandbox:/mnt/data/rl_stagewise_patch.diff)

The best next move is to run `kaggle_medium`, then select between `checkpoints/stagewise/stage*.pt` and `checkpoints/stagewise/best_stage*.pt` using the 50-match timeout benchmark.
