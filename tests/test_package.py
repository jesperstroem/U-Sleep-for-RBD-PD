"""Structural tests that run in CI without requiring private GitLab dependencies."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


class TestWeights:
    def test_pretrained_model_exists(self):
        assert (REPO_ROOT / "weights" / "Pretrained_Model.ckpt").exists(), (
            "Pretrained_Model.ckpt not found in weights/. Make sure the weights are committed to the repository."
        )

    def test_generalized_model_exists(self):
        assert (REPO_ROOT / "weights" / "Generalized_Model.ckpt").exists(), (
            "Generalized_Model.ckpt not found in weights/. Make sure the weights are committed to the repository."
        )


class TestPackaging:
    def test_pyproject_toml_exists(self):
        assert (REPO_ROOT / "pyproject.toml").exists()

    def test_pyproject_toml_is_valid_toml(self):
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            tomllib = pytest.importorskip("tomli", reason="tomli required on Python <3.11")

        content = (REPO_ROOT / "pyproject.toml").read_bytes()
        data = tomllib.loads(content.decode())
        assert "project" in data
        assert data["project"]["name"] == "usleep-rbdpd"

    def test_no_setup_py(self):
        assert not (REPO_ROOT / "setup.py").exists(), (
            "setup.py still exists — remove it now that pyproject.toml is in place."
        )

    def test_readme_exists(self):
        assert (REPO_ROOT / "README.md").exists()


class TestNotebook:
    def _load_notebook(self) -> dict:
        return json.loads((REPO_ROOT / "demo.ipynb").read_text(encoding="utf-8"))

    def test_notebook_exists(self):
        assert (REPO_ROOT / "demo.ipynb").exists()

    def test_notebook_is_valid_json(self):
        nb = self._load_notebook()
        assert "cells" in nb
        assert "nbformat" in nb

    def test_notebook_has_code_cells(self):
        nb = self._load_notebook()
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 3, "Expected at least 3 code cells in demo.ipynb"

    def test_notebook_imports_matplotlib(self):
        nb = self._load_notebook()
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        first_code = "".join(code_cells[0]["source"])
        assert "matplotlib" in first_code, (
            "matplotlib is not imported in the first code cell. "
            "Add 'import matplotlib.pyplot as plt' to the imports cell."
        )

    def test_notebook_imports_score_file(self):
        nb = self._load_notebook()
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        first_code = "".join(code_cells[0]["source"])
        assert "score_file" in first_code
        assert "csdp_training.experiments.bids_predictor_single_file" in first_code

    def test_notebook_imports_plot_hypnogram(self):
        nb = self._load_notebook()
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        first_code = "".join(code_cells[0]["source"])
        assert "plotHypnoGram" in first_code
        assert "csdp_pipeline.pipeline_elements.plot_hypnogram" in first_code
