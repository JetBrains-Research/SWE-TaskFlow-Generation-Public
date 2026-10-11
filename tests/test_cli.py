import json

from datasets import Dataset
from typer.testing import CliRunner

from interactive_benchmark_generation.cli import app
from interactive_benchmark_generation.llm import SplitResult


def test_generate_cli_writes_dataset_without_real_llm(monkeypatch, tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "out"
    dataset = Dataset.from_list(
        [
            {
                "instance_id": "repo__issue-1",
                "problem_statement": "Fix the nested model composition bug.",
            }
        ]
    )

    monkeypatch.setattr("interactive_benchmark_generation.cli.load_source_dataset", lambda _name, _split: dataset)
    monkeypatch.setattr(
        "interactive_benchmark_generation.cli.split_problem_statement",
        lambda **_kwargs: SplitResult(
            parts=[
                "Fix the nested model composition bug.",
                "The issue is about nested model composition.",
            ],
            rationale="test",
            coverage_check="ok",
            validation_issues=[],
        ),
    )

    result = runner.invoke(
        app,
        [
            "generate",
            "--output-dir",
            str(output_dir),
            "--preset",
            "swe-bench-verified",
            "--limit",
            "1",
            "--shuffle-seed",
            "123",
            "--n-parts",
            "2",
            "--workers",
            "2",
            "--env-file",
            str(tmp_path / "missing.env"),
            "--litellm-base-url",
            "",
            "--no-llm-validation",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "dataset").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "checkpoint" / "manifest.json").exists()
    assert (output_dir / "checkpoint" / "results" / "00000000.json").exists()
    assert json.loads((output_dir / "metadata.json").read_text())["workers"] == 2
