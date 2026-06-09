import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "colab" / "stagewise_rl_training.ipynb"
ROOT = Path(__file__).resolve().parents[1]


def _sources():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(cell.get("source", [])) for cell in nb["cells"]]


def test_notebook_is_valid_json():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 14


def test_notebook_has_required_sections():
    text = "\n".join(_sources())
    required = [
        "stage_train_bc_ppo.py",
        "Behavior Cloning",
        "PPO training",
        "benchmark",
        "--eval-matches",
    ]
    for marker in required:
        assert marker in text


def test_notebook_uses_stagewise_approach():
    text = "\n".join(_sources())
    assert "kaggle_medium" in text
    assert "--profile" in text
    assert "EVAL_MATCHES" in text


def test_notebook_applies_patches():
    text = "\n".join(_sources())
    assert "patch" in text and ".diff" in text


def test_notebook_is_gpu_enabled():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["metadata"].get("accelerator") == "GPU"


def test_train_bc_ppo_script_has_cli_entrypoint():
    script = ROOT / "scripts" / "participant" / "train_bc_ppo.py"
    text = script.read_text()
    assert "def main(" in text
    assert "argparse.ArgumentParser" in text
    assert 'if __name__ == "__main__":' in text


def test_stage_train_script_exists():
    assert (ROOT / "scripts" / "participant" / "stage_train_bc_ppo.py").exists()


def test_curriculum_configs_exist():
    assert (ROOT / "configs" / "curriculum_stagewise_v2.json").exists()
    assert (ROOT / "configs" / "curriculum_sandbox_stagewise.json").exists()


def test_notebook_reflects_new_layout():
    text = "\n".join(_sources())
    assert "docs/patches" in text
    assert "Continuous Curriculum" in text
    assert "opponent_schedule" in text
    assert "MILESTONE_DIR" in text
