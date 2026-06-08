# BC + PPO Colab Notebook Design

## Goal

Create `colab/base.ipynb` as a self-contained Colab notebook for training a hybrid Bomberland agent with behavior cloning warm start and PPO fine-tuning.

## Scope

The notebook must run from Colab against this repository, use GPU for PyTorch training when available, and export a CPU-compatible submission bundle containing `agent.py` and `model.pt`.

## Architecture

The notebook implements four layers:

1. Repository setup and imports.
2. Shared game utilities: observation encoding, legal action masks, danger maps, BFS escape, bomb veto, reward shaping, and ranking reward.
3. Training pipeline: expert trajectory collection, behavior cloning, PPO rollout/update, snapshot opponent support, and metrics.
4. Export and smoke tests: write a submission folder, load it through the repo runtime loader, run short matches, and optionally benchmark.

## Model

Use a compact CNN + MLP actor-critic:

- Spatial input: 13 x 13 feature planes for terrain, items, player positions, bombs, timers, ownership, danger horizon, reachable cells, bomb utility, and safe cells.
- Scalar input: agent id, step fraction, bombs left, radius, alive opponent count, nearest enemy/item/box spot distance, current danger, escape-after-bomb, boxes hit, and enemies threatened.
- Outputs: 6 policy logits and 1 scalar value.

## Training

Behavior cloning collects actions from `agent/codex/1.py`, `agent/codex/2.py`, `TacticalRuleAgent`, `GeniusRuleAgent`, `SmarterRuleAgent`, and `BoxFarmerAgent`. Invalid expert labels after masking are skipped.

PPO fine-tuning starts from the BC model. Rollouts sample the controlled agent id, use a baseline/snapshot opponent pool, apply legal masks and safety overrides, and optimize clipped PPO loss with GAE.

## Safety

The safety layer must be used in both training and export:

- Mask illegal movement, blocked bomb actions, and unsafe bombs.
- Force BFS escape when the current tile is threatened.
- Veto bomb placement unless it has value and an escape path.
- Fall back to safe movement or `STOP` if inference fails.

## Export

The notebook writes:

- `exports/hybrid_ppo_agent/agent.py`
- `exports/hybrid_ppo_agent/model.pt`

The exported agent must load on CPU, call `torch.inference_mode()`, preserve the engine action ids, and avoid network or file writes during `act()`.

## Verification

Verification covers:

- `colab/base.ipynb` is valid notebook JSON.
- Notebook contains setup, BC, PPO, export, and smoke-test sections.
- Exported code template contains `class Agent`, `PolicyValueNet`, and `torch.inference_mode()`.
- Optional runtime smoke tests can be run from the notebook with tiny settings.
