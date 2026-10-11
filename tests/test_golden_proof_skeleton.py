"""The proof skeleton handed to the generator must satisfy the proof validator.

The structural rules for a golden proof are enforced by regexes in
`_validate_golden_proof_source` but were only ever described to the model in
prose. Sol happened to write an accepted spelling every time; Terra did not, and
222 issues in the first full Terra run died at ingestion for it.

The prompt now shows this skeleton literally. That only helps while the skeleton
is genuinely acceptable, so it is checked here against the validator itself
rather than against a copy of its expectations.
"""

from __future__ import annotations

from pathlib import Path

from interactive_benchmark_generation.qa_io import _validate_golden_proof_source

SKELETON = (
    Path(__file__).resolve().parents[1]
    / "src/interactive_benchmark_generation/templates/golden_proof_skeleton.sh"
)


def test_the_skeleton_passes_structural_preflight():
    report = _validate_golden_proof_source(SKELETON.read_text(), "q-001.sh")
    assert report["asserted_source_files"] == ["pkg/module_a.py", "pkg/module_b.py"]
    assert report["required_tools"] == ["python3"]
    assert report["python_bytecode_disabled"] is True
    assert report["repository_write_commands"] == []


def test_the_skeleton_demonstrates_each_rule_the_validator_enforces():
    text = SKELETON.read_text()
    for fragment in (
        "set -euo pipefail",
        "PROOF_REQUIRED_TOOLS=(",
        "command -v",
        "missing required proof tool",
        'REPO_ROOT/$source_file',
        'export PYTHONPATH="$REPO_ROOT"',
        "PYTHONNOUSERSITE=1",
        "PYTHONSAFEPATH=1",
        "PYTHONDONTWRITEBYTECODE=1",
        'cd "$REPO_ROOT"',
        "qa_proof/v1",
        "mktemp -d",
    ):
        assert fragment in text, fragment
