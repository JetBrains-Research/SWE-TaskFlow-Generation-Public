import json
import time

import pytest
from datasets import Dataset

from interactive_benchmark_generation.dataset_io import (
    build_task_description,
    generate_records,
    load_source_dataset,
    save_generated_dataset,
    select_limit,
)
from interactive_benchmark_generation.llm import EligibilityResult, IneligibleProblemError, SplitResult


def test_build_task_description_includes_swe_bench_pro_requirements_and_interface():
    description = build_task_description(
        {
            "problem_statement": "Add mapping union support.",
            "requirements": "Support normal and in-place union.",
            "interface": "Add `__or__`, `__ror__`, and `__ior__`.",
        }
    )

    assert description == (
        "Add mapping union support.\n\n"
        "## Requirements\n\n"
        "Support normal and in-place union.\n\n"
        "## Interface\n\n"
        "Add `__or__`, `__ror__`, and `__ior__`."
    )


def test_build_task_description_decodes_quoted_swe_bench_pro_fields():
    description = build_task_description(
        {
            "problem_statement": '"Implement union support."',
            "requirements": '"Support `|` and `|=`."',
            "interface": '"No new interfaces are introduced."',
        }
    )

    assert description == (
        "Implement union support.\n\n"
        "## Requirements\n\n"
        "Support `|` and `|=`.\n\n"
        "## Interface\n\n"
        "No new interfaces are introduced."
    )


def test_generate_records_adds_problem_statements():
    dataset = Dataset.from_list(
        [
            {
                "instance_id": "repo__issue-1",
                "problem_statement": "Fix the rendering bug in nested charts.",
            }
        ]
    )

    def splitter(problem_statement: str) -> SplitResult:
        return SplitResult(
            parts=[problem_statement, "Nested charts are involved."],
            rationale="test",
            coverage_check="ok",
            validation_issues=[],
        )

    records, debug = generate_records(dataset, splitter)

    assert records[0]["problem_statements"] == [
        "Fix the rendering bug in nested charts.",
        "Nested charts are involved.",
    ]
    assert debug[0]["instance_id"] == "repo__issue-1"


def test_generate_records_skips_rejected_instances():
    dataset = Dataset.from_list(
        [
            {
                "instance_id": "repo__accepted",
                "problem_statement": "Keep this task.",
            },
            {
                "instance_id": "repo__rejected",
                "problem_statement": "Reject this task.",
            },
        ]
    )

    def splitter(problem_statement: str) -> SplitResult:
        if problem_statement == "Reject this task.":
            raise ValueError("Could not generate a valid split")
        return SplitResult(
            parts=["Keep this task actionable.", "Preserve all of its information."],
            rationale="test",
            coverage_check="ok",
            validation_issues=[],
        )

    records, debug = generate_records(dataset, splitter)

    assert [record["instance_id"] for record in records] == ["repo__accepted"]
    assert debug[0]["status"] == "accepted"
    assert debug[1]["status"] == "rejected"
    assert debug[1]["validation_issues"] == ["Could not generate a valid split"]


def test_generate_records_filters_atomic_instances():
    dataset = Dataset.from_list(
        [
            {
                "instance_id": "repo__atomic",
                "problem_statement": "One behavior is broken.",
            }
        ]
    )

    def splitter(_problem_statement: str) -> SplitResult:
        raise IneligibleProblemError(
            EligibilityResult(
                eligible=False,
                category="atomic_evidence_only",
                units=(),
                reason="Only one acceptance condition is present.",
            )
        )

    records, debug = generate_records(dataset, splitter)

    assert records == []
    assert debug[0]["status"] == "filtered"
    assert debug[0]["filter_category"] == "atomic_evidence_only"
    assert debug[0]["filter_reason"] == "Only one acceptance condition is present."


def test_generate_records_preserves_source_order_with_parallel_workers():
    dataset = Dataset.from_list(
        [
            {"instance_id": "slow", "problem_statement": "Slow task."},
            {"instance_id": "fast", "problem_statement": "Fast task."},
        ]
    )

    def splitter(problem_statement: str) -> SplitResult:
        if problem_statement == "Slow task.":
            time.sleep(0.05)
        return SplitResult(
            parts=[problem_statement, "More details."],
            rationale="test",
            coverage_check="ok",
            validation_issues=[],
        )

    records, debug = generate_records(dataset, splitter, workers=2)

    assert [record["instance_id"] for record in records] == ["slow", "fast"]
    assert [item["instance_id"] for item in debug] == ["slow", "fast"]


def test_generate_records_checkpoints_and_resumes_missing_instances(tmp_path):
    dataset = Dataset.from_list(
        [
            {"instance_id": "first", "problem_statement": "First task."},
            {"instance_id": "second", "problem_statement": "Second task."},
        ]
    )
    output_dir = tmp_path / "generated"

    def interrupted_splitter(problem_statement: str) -> SplitResult:
        if problem_statement == "Second task.":
            raise RuntimeError("interrupted")
        return SplitResult(
            parts=[problem_statement, "More details."],
            rationale="test",
            coverage_check="ok",
            validation_issues=[],
        )

    with pytest.raises(RuntimeError, match="interrupted"):
        generate_records(
            dataset,
            interrupted_splitter,
            workers=1,
            output_dir=output_dir,
            checkpoint_config={"model": "test"},
        )

    calls = []

    def resumed_splitter(problem_statement: str) -> SplitResult:
        calls.append(problem_statement)
        return SplitResult(
            parts=[problem_statement, "More details."],
            rationale="test",
            coverage_check="ok",
            validation_issues=[],
        )

    records, debug = generate_records(
        dataset,
        resumed_splitter,
        workers=2,
        output_dir=output_dir,
        resume=True,
        checkpoint_config={"model": "test"},
    )

    assert calls == ["Second task."]
    assert [record["instance_id"] for record in records] == ["first", "second"]
    assert [item["status"] for item in debug] == ["accepted", "accepted"]
    assert len(list((output_dir / "checkpoint" / "results").glob("*.json"))) == 2


def test_generate_records_rejects_mismatched_resume(tmp_path):
    dataset = Dataset.from_list([{"instance_id": "one", "problem_statement": "One task."}])

    def splitter(problem_statement: str) -> SplitResult:
        return SplitResult(
            parts=[problem_statement, "More details."],
            rationale="test",
            coverage_check="ok",
            validation_issues=[],
        )

    generate_records(
        dataset,
        splitter,
        output_dir=tmp_path,
        checkpoint_config={"model": "first"},
    )

    with pytest.raises(ValueError, match="does not match"):
        generate_records(
            dataset,
            splitter,
            output_dir=tmp_path,
            resume=True,
            checkpoint_config={"model": "second"},
        )


def test_generate_records_resume_keeps_filtered_and_rejected_results(tmp_path):
    dataset = Dataset.from_list(
        [
            {"instance_id": "filtered", "problem_statement": "Filtered task."},
            {"instance_id": "rejected", "problem_statement": "Rejected task."},
        ]
    )

    def initial_splitter(problem_statement: str) -> SplitResult:
        if problem_statement == "Filtered task.":
            raise IneligibleProblemError(
                EligibilityResult(
                    eligible=False,
                    category="atomic_evidence_only",
                    units=(),
                    reason="Not enough units.",
                )
            )
        raise ValueError("Invalid split")

    generate_records(
        dataset,
        initial_splitter,
        output_dir=tmp_path,
        checkpoint_config={"model": "test"},
    )

    def unexpected_splitter(_problem_statement: str) -> SplitResult:
        raise AssertionError("Completed result was retried")

    records, debug = generate_records(
        dataset,
        unexpected_splitter,
        output_dir=tmp_path,
        resume=True,
        checkpoint_config={"model": "test"},
    )

    assert records == []
    assert [item["status"] for item in debug] == ["filtered", "rejected"]


def test_generate_records_rejects_corrupt_checkpoint(tmp_path):
    dataset = Dataset.from_list([{"instance_id": "one", "problem_statement": "One task."}])

    def splitter(problem_statement: str) -> SplitResult:
        return SplitResult(
            parts=[problem_statement, "More details."],
            rationale="test",
            coverage_check="ok",
            validation_issues=[],
        )

    generate_records(
        dataset,
        splitter,
        output_dir=tmp_path,
        checkpoint_config={"model": "test"},
    )
    result_path = tmp_path / "checkpoint" / "results" / "00000000.json"
    result_path.write_text("{")

    with pytest.raises(ValueError, match="Invalid checkpoint result"):
        generate_records(
            dataset,
            splitter,
            output_dir=tmp_path,
            resume=True,
            checkpoint_config={"model": "test"},
        )


def test_save_generated_dataset_round_trip(tmp_path):
    records = [
        {
            "instance_id": "repo__issue-1",
            "problem_statement": "Fix the bug.",
            "problem_statements": ["Fix the bug.", "The failure is in nested handling."],
        }
    ]
    output_dir = tmp_path / "generated"

    dataset_dir = save_generated_dataset(
        records=records,
        debug=[],
        output_dir=output_dir,
        metadata={"source_dataset": "local", "split": "test"},
    )
    loaded = load_source_dataset(str(dataset_dir), "train")
    metadata = json.loads((output_dir / "metadata.json").read_text())

    assert loaded[0]["problem_statements"] == ["Fix the bug.", "The failure is in nested handling."]
    assert (output_dir / "preview.jsonl").exists()
    prompts_markdown = (output_dir / "prompts.md").read_text()
    assert "### Single prompt" in prompts_markdown
    assert "### Multi prompt" in prompts_markdown
    assert json.loads((output_dir / "preview.json").read_text())[0]["instance_id"] == "repo__issue-1"
    assert metadata["n_instances"] == 1
    assert metadata["n_filtered_instances"] == 0
    assert metadata["n_rejected_instances"] == 0


def test_select_limit_can_shuffle_before_limiting():
    dataset = Dataset.from_list([{"value": value} for value in range(20)])

    first = select_limit(dataset, limit=5, shuffle_seed=123)
    second = select_limit(dataset, limit=5, shuffle_seed=123)

    assert first["value"] == second["value"]
    assert first["value"] != [0, 1, 2, 3, 4]
