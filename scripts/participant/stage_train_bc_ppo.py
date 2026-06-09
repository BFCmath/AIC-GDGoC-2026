"""Stage-wise BC+PPO training driver for Kaggle/Colab.

This is a practical orchestrator around train_bc_ppo.py.  It stores a BC
checkpoint, then trains fixed curriculum stages one by one.  Each stage can use
its own PPO LR/entropy/update budget and writes stage checkpoints so you can
resume from the best observed stage.

Example quick GPU run:
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
    # Fast smoke: intended for CPU/sandbox debugging.
    "smoke": {
        "bc_matches": 1,
        "bc_epochs": 1,
        "bc_max_steps": 40,
        "curriculum": "configs/curriculum_sandbox_stagewise.json",
        "stages": [
            {"stage": 0, "updates": 2, "envs": 2, "horizon": 16, "lr": 3.0e-4, "ent": 0.020},
            {"stage": 1, "updates": 2, "envs": 2, "horizon": 16, "lr": 2.5e-4, "ent": 0.018},
        ],
    },
    # Reasonable first Kaggle pass, usually minutes rather than hours.
    "kaggle_medium": {
        "bc_matches": 50,
        "bc_epochs": 28,
        "bc_max_steps": 220,
        "curriculum": "configs/curriculum_stagewise_v2.json",
        "stages": [
            {"stage": 0, "updates": 30, "envs": 16, "horizon": 128, "lr": 3.0e-4, "ent": 0.020},
            {"stage": 1, "updates": 40, "envs": 16, "horizon": 128, "lr": 2.5e-4, "ent": 0.018},
            {"stage": 2, "updates": 50, "envs": 16, "horizon": 160, "lr": 2.0e-4, "ent": 0.016},
            {"stage": 3, "updates": 60, "envs": 16, "horizon": 192, "lr": 1.6e-4, "ent": 0.014},
            {"stage": 4, "updates": 70, "envs": 16, "horizon": 224, "lr": 1.2e-4, "ent": 0.012},
            {"stage": 5, "updates": 80, "envs": 16, "horizon": 256, "lr": 1.0e-4, "ent": 0.010},
            {"stage": 6, "updates": 90, "envs": 16, "horizon": 256, "lr": 8.0e-5, "ent": 0.008},
            {"stage": 7, "updates": 120, "envs": 16, "horizon": 256, "lr": 6.0e-5, "ent": 0.006},
        ],
    },
    # Longer run for 2xT4/P100.  Use when the medium run is stable.
    "kaggle_long": {
        "bc_matches": 120,
        "bc_epochs": 45,
        "bc_max_steps": 300,
        "curriculum": "configs/curriculum_stagewise_v2.json",
        "stages": [
            {"stage": 0, "updates": 60, "envs": 24, "horizon": 160, "lr": 3.0e-4, "ent": 0.020},
            {"stage": 1, "updates": 80, "envs": 24, "horizon": 192, "lr": 2.5e-4, "ent": 0.018},
            {"stage": 2, "updates": 100, "envs": 24, "horizon": 224, "lr": 2.0e-4, "ent": 0.015},
            {"stage": 3, "updates": 120, "envs": 24, "horizon": 256, "lr": 1.5e-4, "ent": 0.013},
            {"stage": 4, "updates": 140, "envs": 24, "horizon": 256, "lr": 1.2e-4, "ent": 0.011},
            {"stage": 5, "updates": 160, "envs": 24, "horizon": 320, "lr": 9.0e-5, "ent": 0.009},
            {"stage": 6, "updates": 180, "envs": 24, "horizon": 320, "lr": 7.0e-5, "ent": 0.007},
            {"stage": 7, "updates": 240, "envs": 24, "horizon": 384, "lr": 5.0e-5, "ent": 0.005},
        ],
    },
}


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="kaggle_medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=86)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--skip-bc", action="store_true")
    ap.add_argument("--start-stage", type=int, default=0)
    ap.add_argument("--end-stage", type=int, default=7)
    ap.add_argument("--eval-matches", type=int, default=12)
    ap.add_argument("--export-dir", default="exports/stagewise_hybrid_agent")
    ap.add_argument("--curriculum-config", help="Override curriculum config JSON path")
    ap.add_argument("--overrides", help="Path to overrides JSON (stages, curriculum, etc.)")
    args = ap.parse_args()
    cfg = PROFILES[args.profile]
    curriculum_path = args.curriculum_config or str(ROOT / cfg["curriculum"])
    
    if args.overrides:
        overrides_path = Path(args.overrides)
        if not overrides_path.is_absolute():
            overrides_path = ROOT / overrides_path
        if overrides_path.exists():
            overrides = json.loads(overrides_path.read_text())
            stage_overrides = overrides.get("stages", {})
            for st in cfg["stages"]:
                key = str(st["stage"])
                if key in stage_overrides:
                    st.update(stage_overrides[key])
                    print(f"Override stage {key}: {stage_overrides[key]}")
            curr_overrides = overrides.get("curriculum", {})
            if curr_overrides:
                base = json.loads(Path(ROOT / cfg["curriculum"]).read_text())
                for s, vals in curr_overrides.items():
                    if s in base:
                        base[s].update(vals)
                    else:
                        base[s] = vals
                custom_path = ROOT / "configs" / "_generated_curriculum.json"
                custom_path.write_text(json.dumps(base, indent=2))
                curriculum_path = str(custom_path)
                print(f"Override curriculum -> {custom_path}")
        else:
            print(f"Warning: overrides file {overrides_path} not found")

    ckpt_dir = ROOT / "checkpoints/stagewise"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bc_ckpt = ckpt_dir / "bc.pt"
    current = bc_ckpt

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

    for st in cfg["stages"]:
        stage = int(st["stage"])
        if stage < args.start_stage or stage > args.end_stage:
            continue
        out = ckpt_dir / f"stage{stage}.pt"
        ppo_flags = [
            sys.executable, "-u", str(TRAIN), "--mode", "ppo", "--device", args.device,
            "--seed", str(args.seed + 1000 * stage), "--torch-threads", str(args.torch_threads),
            "--checkpoint", str(current), "--save-checkpoint", str(out),
            "--curriculum-config", curriculum_path,
            "--fixed-stage", str(stage),
            "--ppo-updates", str(st["updates"]),
            "--ppo-envs-per-update", str(st["envs"]),
            "--ppo-horizon", str(st["horizon"]),
            "--ppo-epochs", "4", "--ppo-minibatch-size", "512",
            "--ppo-lr", str(st["lr"]), "--ent-coef", str(st["ent"]),
            "--eval-interval", "5", "--eval-matches", str(args.eval_matches),
            "--snapshot-interval", "5",
            "--stage-checkpoint-dir", str(ckpt_dir / "snapshots"),
            "--best-checkpoint", str(ckpt_dir / f"best_stage{stage}.pt"),
        ]
        if args.overrides:
            ppo_flags.extend(["--overrides", args.overrides])
        run(ppo_flags)
        current = out

    run([
        sys.executable, "-u", str(TRAIN), "--mode", "export", "--device", "cpu",
        "--checkpoint", str(current), "--export-dir", args.export_dir,
    ])

    print("\nStage-wise training completed.")
    print(f"Final checkpoint: {current}")
    print(f"Export folder: {ROOT / args.export_dir}")


if __name__ == "__main__":
    main()
