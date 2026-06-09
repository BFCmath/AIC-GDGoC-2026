"""Continuous curriculum BC+PPO training driver for Kaggle/Colab.

Orchestrates behavior cloning, a single continuous PPO training run with
smooth opponent priority scheduling, hyperparameter annealing, milestone
checkpoints, and final export.

Example:
    python scripts/participant/stage_train_bc_ppo.py --device cuda --profile kaggle_medium
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts/participant/train_bc_ppo.py"

PROFILES = {
    "smoke": {
        "bc_matches": 1,
        "bc_epochs": 1,
        "bc_max_steps": 40,
        "ppo_updates": 10,
        "ppo_envs": 16,
        "ppo_horizon_start": 32,
        "ppo_horizon_end": 32,
        "ppo_lr_start": 3.0e-4,
        "ppo_lr_end": 3.0e-4,
        "ppo_ent_start": 0.020,
        "ppo_ent_end": 0.020,
        "max_steps_start": 100,
        "max_steps_end": 100,
    },
    "kaggle_medium": {
        "bc_matches": 80,
        "bc_epochs": 35,
        "bc_max_steps": 300,
        "ppo_updates": 900,
        "ppo_envs": 16,
        "ppo_horizon_start": 128,
        "ppo_horizon_end": 256,
        "ppo_lr_start": 3.0e-4,
        "ppo_lr_end": 6.0e-5,
        "ppo_ent_start": 0.018,
        "ppo_ent_end": 0.006,
        "max_steps_start": 160,
        "max_steps_end": 500,
    },
    "kaggle_long": {
        "bc_matches": 150,
        "bc_epochs": 45,
        "bc_max_steps": 350,
        "ppo_updates": 1600,
        "ppo_envs": 16,
        "ppo_horizon_start": 128,
        "ppo_horizon_end": 256,
        "ppo_lr_start": 3.0e-4,
        "ppo_lr_end": 5.0e-5,
        "ppo_ent_start": 0.020,
        "ppo_ent_end": 0.005,
        "max_steps_start": 160,
        "max_steps_end": 500,
    },
}

DEFAULT_PPO_ARGS = [
    "--ppo-epochs", "4",
    "--ppo-minibatch-size", "512",
    "--eval-interval", "10",
    "--eval-matches", "16",
    "--snapshot-interval", "10",
    "--gamma", "0.995",
    "--gae-lambda", "0.95",
    "--clip-eps", "0.2",
    "--vf-coef", "0.5",
    "--grad-clip", "0.5",
]


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="kaggle_medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=86)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--skip-bc", action="store_true")
    ap.add_argument("--resume", default="", help="Checkpoint to resume PPO from (skips BC)")
    ap.add_argument("--export-dir", default="exports/stagewise_hybrid_agent")
    ap.add_argument("--eval-matches", type=int, default=None, help="Override eval matches for PPO")
    ap.add_argument("--overrides", help="Path to overrides JSON (opponent_schedule, reward, hyperparam)")
    ap.add_argument("--milestone-dir", default="checkpoints/milestones")
    args = ap.parse_args()
    cfg = PROFILES[args.profile]

    ckpt_dir = ROOT / args.milestone_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bc_ckpt = ckpt_dir / "bc.pt"
    final_ckpt = ckpt_dir / "final.pt"

    if args.resume:
        args.skip_bc = True

    if not args.skip_bc:
        run([
            sys.executable, "-u", str(TRAIN), "--mode", "bc", "--device", args.device,
            "--seed", str(args.seed), "--torch-threads", str(args.torch_threads),
            "--max-steps", str(cfg["bc_max_steps"]),
            "--bc-matches", str(cfg["bc_matches"]),
            "--bc-epochs", str(cfg["bc_epochs"]),
            "--bc-batch-size", "512", "--bc-lr", "3e-4",
            "--save-checkpoint", str(bc_ckpt),
        ])
    elif not bc_ckpt.exists():
        raise FileNotFoundError(f"--skip-bc requested but {bc_ckpt} does not exist")

    ppo_start_ckpt = Path(args.resume) if args.resume else bc_ckpt
    ppo_flags = [
        sys.executable, "-u", str(TRAIN), "--mode", "ppo", "--device", args.device,
        "--seed", str(args.seed), "--torch-threads", str(args.torch_threads),
        "--checkpoint", str(ppo_start_ckpt), "--save-checkpoint", str(final_ckpt),
        "--ppo-updates", str(cfg["ppo_updates"]),
        "--ppo-envs-per-update", str(cfg["ppo_envs"]),
        "--ppo-horizon", str(cfg.get("ppo_horizon_start", 128)),
        "--ppo-horizon-start", str(cfg["ppo_horizon_start"]),
        "--ppo-horizon-end", str(cfg["ppo_horizon_end"]),
        "--ppo-lr-start", str(cfg["ppo_lr_start"]),
        "--ppo-lr-end", str(cfg["ppo_lr_end"]),
        "--ppo-ent-start", str(cfg["ppo_ent_start"]),
        "--ppo-ent-end", str(cfg["ppo_ent_end"]),
        "--max-steps-start", str(cfg["max_steps_start"]),
        "--max-steps-end", str(cfg["max_steps_end"]),
        "--stage-checkpoint-dir", str(ckpt_dir),
        "--best-checkpoint", str(ckpt_dir / "best.pt"),
    ] + DEFAULT_PPO_ARGS

    if args.eval_matches is not None:
        ppo_flags.extend(["--eval-matches", str(args.eval_matches)])
    if args.overrides:
        ppo_flags.extend(["--overrides", args.overrides])

    run(ppo_flags)

    export_ckpt = ckpt_dir / "best.pt"
    if not export_ckpt.exists():
        export_ckpt = final_ckpt
    run([
        sys.executable, "-u", str(TRAIN), "--mode", "export", "--device", "cpu",
        "--checkpoint", str(export_ckpt), "--export-dir", args.export_dir,
    ])

    print("\nContinuous curriculum training completed.")
    print(f"Best checkpoint:  {ckpt_dir / 'best.pt'}")
    print(f"Final checkpoint: {final_ckpt}")
    print(f"Exported from:    {export_ckpt}")
    print(f"Export folder:    {ROOT / args.export_dir}")
    print(f"Milestones:       {ckpt_dir}")


if __name__ == "__main__":
    main()
