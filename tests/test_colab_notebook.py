import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "colab" / "base.ipynb"
ROOT = Path(__file__).resolve().parents[1]


def _sources():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(cell.get("source", [])) for cell in nb["cells"]]


def test_colab_base_is_valid_notebook_json():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10


def test_colab_base_contains_bc_ppo_export_sections():
    text = "\n".join(_sources())
    required = [
        "Bomberland BC + PPO Hybrid Training",
        "collect_expert_dataset",
        "train_behavior_cloning",
        "train_ppo",
        "export_agent",
        "exports/hybrid_ppo_agent",
    ]
    for marker in required:
        assert marker in text


def test_colab_base_clones_bfcmath_training_repo_and_calls_script():
    text = "\n".join(_sources())
    assert "https://github.com/BFCmath/AIC-GDGoC-2026.git" in text
    assert "scripts/participant/train_bc_ppo.py" in text
    assert "--mode full" in text


def test_export_template_contains_submission_agent_contract():
    text = "\n".join(_sources())
    assert "class Agent:" in text
    assert "class PolicyValueNet" in text
    assert "torch.inference_mode()" in text


def test_train_bc_ppo_script_has_cli_entrypoint():
    script = ROOT / "scripts" / "participant" / "train_bc_ppo.py"
    text = script.read_text()
    assert "def main(" in text
    assert "argparse.ArgumentParser" in text
    assert "if __name__ == \"__main__\":" in text
