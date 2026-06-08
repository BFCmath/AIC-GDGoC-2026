# BC + PPO Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab notebook that trains a hybrid Bomberland agent with behavior cloning and PPO, then exports a submission-ready bundle.

**Architecture:** Keep training code self-contained in `colab/base.ipynb` for Colab portability. Add lightweight tests that validate notebook structure and key exported-code invariants without running expensive RL training.

**Tech Stack:** Python, Jupyter notebook JSON, PyTorch, NumPy, repo `engine` and `agent` modules, pytest.

---

## File Structure

- Modify `colab/base.ipynb`: self-contained BC + PPO training notebook.
- Create `tests/test_colab_notebook.py`: validates notebook JSON and required cells/templates.
- Create `docs/superpowers/specs/2026-06-09-bc-ppo-colab-design.md`: design record.
- Create `docs/superpowers/plans/2026-06-09-bc-ppo-colab.md`: implementation plan.

## Tasks

### Task 1: Notebook Structure Test

**Files:**
- Create: `tests/test_colab_notebook.py`

- [ ] **Step 1: Write failing tests**

```python
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "colab" / "base.ipynb"


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


def test_export_template_contains_submission_agent_contract():
    text = "\n".join(_sources())
    assert "class Agent:" in text
    assert "class PolicyValueNet" in text
    assert "torch.inference_mode()" in text
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_colab_notebook.py -q`

Expected: fails because `colab/base.ipynb` is empty.

### Task 2: Generate Notebook

**Files:**
- Modify: `colab/base.ipynb`

- [ ] **Step 1: Write notebook JSON**

Create notebook cells for setup, configuration, imports, safety utilities, encoder, model, expert collection, BC training, PPO training, export, and smoke tests.

- [ ] **Step 2: Run notebook structure tests**

Run: `pytest tests/test_colab_notebook.py -q`

Expected: all tests pass.

### Task 3: Validate Syntax of Embedded Code

**Files:**
- Modify: `colab/base.ipynb`

- [ ] **Step 1: Extract code cells and compile them**

Run:

```bash
python3 - <<'PY'
import ast, json
from pathlib import Path
nb = json.loads(Path("colab/base.ipynb").read_text())
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        ast.parse("".join(cell.get("source", [])), filename=f"cell_{i}")
print("compiled")
PY
```

Expected: prints `compiled`.

### Task 4: Final Verification

**Files:**
- Inspect: `colab/base.ipynb`
- Inspect: `tests/test_colab_notebook.py`

- [ ] **Step 1: Run verification**

Run:

```bash
pytest tests/test_colab_notebook.py -q
python3 - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path("colab/base.ipynb").read_text())
print(len(nb["cells"]))
PY
```

Expected: tests pass and notebook has at least 10 cells.
