import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from interactive_benchmark_generation.cli import app
from interactive_benchmark_generation.qa_multistep_builder import (
    OUTPUT_FILES,
    PRIVATE_FIELDS,
    _validate_public_record,
    build_hybrid_dataset,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))


def proof_execution(question_id: str, source_sha256: str) -> dict:
    return {
        "protocol": "qa_proof_execution/v1",
        "question_id": question_id,
        "proof_kind": "golden",
        "source_sha256": source_sha256,
        "submitted": True,
        "executed": True,
        "success": True,
        "exit_code": 0,
        "timed_out": False,
        "output_truncated": False,
        "repository_state_unchanged": True,
        "evidence_valid": True,
        "repository_dependent": True,
        "deterministic": True,
    }


def qa_question(
    instance_id: str,
    question_id: str,
    proof_sha256: str,
) -> dict:
    return {
        "schema_version": "2.0",
        "instance_id": instance_id,
        "fact_id": f"fact-{question_id}",
        "fact": "Private fact",
        "question_id": question_id,
        "question": f"How does {question_id} work?",
        "golden_answer": {"canonical": "Private answer"},
        "golden_proof": {
            "path": f"golden-proofs/{instance_id}/{question_id}.sh",
            "sha256": proof_sha256,
        },
        "golden_proof_validation": {
            "operationally_valid": True,
            "repository_dependent": True,
            "deterministic": True,
            "repository_state_unchanged": True,
        },
        "adversarial_review": {"accepted": True, "reasons": ["Private"]},
        "evidence_hops": [{"claim": "Private evidence"}],
        "reasoning_chain": [{"claim": "Private reasoning"}],
    }


def create_qa_bundle(
    root: Path,
    issue_name: str,
    instance_id: str,
    question_ids: tuple[str, ...] = ("q-1",),
    execution_hash: str | None = None,
) -> Path:
    bundle = root / issue_name / "filtered"
    script = "#!/usr/bin/env bash\nprintf 'ok\\n'\n"
    proof_sha256 = hashlib.sha256(script.encode()).hexdigest()
    records = [qa_question(instance_id, question_id, proof_sha256) for question_id in question_ids]
    write_jsonl(bundle / "questions.jsonl", records)
    write_json(
        bundle / "metadata.json",
        {"schema_version": "2.0", "instance_id": instance_id},
    )
    executions = [
        proof_execution(
            question_id,
            execution_hash or proof_sha256,
        )
        for question_id in question_ids
    ]
    write_json(
        bundle / "golden-proof-executions.json",
        {"executions": executions},
    )
    for question_id in question_ids:
        proof = bundle / "golden-proofs" / instance_id / f"{question_id}.sh"
        proof.parent.mkdir(parents=True, exist_ok=True)
        proof.write_text(script)
    return bundle


def create_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    multi = tmp_path / "preview.json"
    records = [
        {
            "instance_id": "instance_repo__task-1",
            "problem_statement": "Combined text is not copied.",
            "problem_statements": ["First issue part.", "Second issue part."],
        },
        {
            "instance_id": "instance_repo__task-2",
            "problem_statement": "Another combined text.",
            "problem_statements": ["Only issue part."],
        },
    ]
    write_json(multi, records)
    qa_root = tmp_path / "qa"
    create_qa_bundle(
        qa_root,
        "issue-0001",
        "instance_repo__task-1",
        ("q-1", "q-2"),
    )
    task_root = tmp_path / "tasks"
    for record in records:
        (task_root / record["instance_id"]).mkdir(parents=True)
    return multi, qa_root, task_root


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def collect_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in collect_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in collect_keys(item)}
    return set()


def public_record(
    prompts: list[dict],
    qa_eligible: bool,
) -> dict:
    without_digest = {
        "schema_version": "qa_multistep_prompt/v1",
        "instance_id": "instance_repo__task-1",
        "qa_eligible": qa_eligible,
        "prompts": prompts,
        "source_digests": {
            "multi_prompt_record_sha256": "0" * 64,
            "qa_bundle_sha256": "1" * 64,
        },
    }
    canonical = (
        json.dumps(
            without_digest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return {
        **without_digest,
        "record_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def test_public_validator_accepts_qa_only_record():
    record = public_record(
        [
            {
                "kind": "qa",
                "question_id": "q-1",
                "text": "How does q-1 work?",
            }
        ],
        qa_eligible=True,
    )

    _validate_public_record(record)


def test_public_validator_rejects_empty_prompts():
    record = public_record([], qa_eligible=False)

    with pytest.raises(ValueError, match="non-empty list"):
        _validate_public_record(record)


def test_public_validator_rejects_qa_eligible_mismatch():
    record = public_record(
        [
            {
                "kind": "qa",
                "question_id": "q-1",
                "text": "How does q-1 work?",
            }
        ],
        qa_eligible=False,
    )

    with pytest.raises(
        ValueError,
        match="qa_eligible must match QA prompt availability",
    ):
        _validate_public_record(record)


def test_public_validator_rejects_noncontiguous_issue_parts():
    record = public_record(
        [
            {
                "kind": "qa",
                "question_id": "q-1",
                "text": "How does q-1 work?",
            },
            {
                "kind": "issue",
                "part_index": 1,
                "text": "Second issue part.",
            },
        ],
        qa_eligible=True,
    )

    with pytest.raises(ValueError, match="contiguous from zero"):
        _validate_public_record(record)


def test_builder_left_joins_qa_and_writes_public_artifacts(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    output = tmp_path / "output"

    result = build_hybrid_dataset(
        multi,
        qa_root,
        task_root,
        output,
    )

    assert sorted(path.name for path in output.iterdir()) == sorted(OUTPUT_FILES)
    records = load_jsonl(output / "hybrid-prompts.jsonl")
    assert [record["instance_id"] for record in records] == [
        "instance_repo__task-1",
        "instance_repo__task-2",
    ]
    assert records[0]["qa_eligible"] is True
    assert records[1]["qa_eligible"] is False
    assert [prompt["kind"] for prompt in records[0]["prompts"]] == [
        "qa",
        "qa",
        "issue",
        "issue",
    ]
    assert records[0]["prompts"][0] == {
        "kind": "qa",
        "question_id": "q-1",
        "text": "How does q-1 work?",
    }
    assert records[0]["prompts"][-1] == {
        "kind": "issue",
        "part_index": 1,
        "text": "Second issue part.",
    }
    assert "Combined text is not copied." not in (output / "hybrid-prompts.jsonl").read_text()
    assert collect_keys(records).isdisjoint(PRIVATE_FIELDS)
    assert result["coverage"] == json.loads((output / "coverage.json").read_text())
    assert result["coverage"]["instances_with_qa"] == 1
    assert result["coverage"]["instances_without_qa"] == 1
    assert result["coverage"]["question_count_distribution"] == {
        "0": 1,
        "2": 1,
    }
    assert result["validation"]["checks"]["private_fields_absent"] is True
    metadata = result["metadata"]
    hybrid_bytes = (output / "hybrid-prompts.jsonl").read_bytes()
    assert metadata["output"]["sha256"] == hashlib.sha256(hybrid_bytes).hexdigest()
    multi_source = metadata["sources"]["multi_prompt_records"]
    assert multi_source["name"] == "preview.json"
    assert multi_source["sha256"] == (hashlib.sha256(multi.read_bytes()).hexdigest())


def test_builder_preserves_prompt_whitespace(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    records = json.loads(multi.read_text())
    records[0]["problem_statements"][0] = "\nFirst issue part.\n"
    write_json(multi, records)
    output = tmp_path / "output"

    build_hybrid_dataset(multi, qa_root, task_root, output)

    hybrid = load_jsonl(output / "hybrid-prompts.jsonl")
    issue_prompts = [prompt for prompt in hybrid[0]["prompts"] if prompt["kind"] == "issue"]
    assert issue_prompts[0]["text"] == "\nFirst issue part.\n"


def test_builder_accepts_legacy_trusted_execution_without_evidence_field(
    tmp_path,
):
    multi, qa_root, task_root = create_sources(tmp_path)
    report_path = qa_root / "issue-0001" / "filtered" / "golden-proof-executions.json"
    report = json.loads(report_path.read_text())
    for execution in report["executions"]:
        execution.pop("evidence_valid")
    write_json(report_path, report)

    build_hybrid_dataset(
        multi,
        qa_root,
        task_root,
        tmp_path / "output",
    )


def test_builder_accepts_combined_proof_execution_report(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    bundle = qa_root / "issue-0001" / "filtered"
    golden_path = bundle / "golden-proof-executions.json"
    report = json.loads(golden_path.read_text())
    report["executions"].append({
        **report["executions"][0],
        "proof_kind": "agent",
    })
    write_json(bundle / "proof-executions.json", report)
    golden_path.unlink()

    result = build_hybrid_dataset(
        multi,
        qa_root,
        task_root,
        tmp_path / "output",
    )

    assert result["coverage"]["instances_with_qa"] == 1


def test_builder_artifacts_are_portable_across_source_roots(tmp_path):
    first_sources = create_sources(tmp_path / "first")
    second_sources = create_sources(tmp_path / "second")
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"

    build_hybrid_dataset(*first_sources, first_output)
    build_hybrid_dataset(*second_sources, second_output)

    assert {path.name: path.read_bytes() for path in first_output.iterdir()} == {
        path.name: path.read_bytes() for path in second_output.iterdir()
    }


def test_builder_verifies_byte_identical_rebuild(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    output = tmp_path / "output"
    build_hybrid_dataset(multi, qa_root, task_root, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    result = build_hybrid_dataset(
        multi,
        qa_root,
        task_root,
        output,
        verify_rebuild=True,
    )

    assert result["verified_rebuild"] is True
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_builder_rejects_non_identical_rebuild(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    output = tmp_path / "output"
    build_hybrid_dataset(multi, qa_root, task_root, output)
    (output / "coverage.json").write_text("{}\n")

    with pytest.raises(
        ValueError,
        match=r"differs byte-for-byte: coverage\.json",
    ):
        build_hybrid_dataset(
            multi,
            qa_root,
            task_root,
            output,
            verify_rebuild=True,
        )


@pytest.mark.parametrize(
    ("records", "error"),
    [
        (
            [
                {
                    "instance_id": "Task",
                    "problem_statements": ["Part"],
                },
                {
                    "instance_id": "task",
                    "problem_statements": ["Part"],
                },
            ],
            "Duplicate multi-prompt instance_id after case-folding",
        ),
        (
            [{"instance_id": "Task", "problem_statements": []}],
            "problem_statements must be a non-empty list",
        ),
        (
            [
                {
                    "instance_id": "Task",
                    "problem_statements": ["Part"],
                    "golden_answer": "secret",
                }
            ],
            "contains private fields",
        ),
    ],
)
def test_builder_rejects_invalid_multi_prompt_records(
    tmp_path,
    records,
    error,
):
    multi = tmp_path / "preview.json"
    write_json(multi, records)
    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    task_root = tmp_path / "tasks"
    task_root.mkdir()

    with pytest.raises(ValueError, match=error):
        build_hybrid_dataset(
            multi,
            qa_root,
            task_root,
            tmp_path / "output",
        )


def test_builder_accepts_jsonl_and_preserves_selected_source_order(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    records = json.loads(multi.read_text())
    jsonl = tmp_path / "preview.jsonl"
    write_jsonl(jsonl, records)

    build_hybrid_dataset(
        jsonl,
        qa_root,
        task_root,
        tmp_path / "output",
        task_names=["instance_repo__task-2", "instance_repo__task-1"],
    )

    assert [record["instance_id"] for record in load_jsonl(tmp_path / "output" / "hybrid-prompts.jsonl")] == [
        "instance_repo__task-1",
        "instance_repo__task-2",
    ]


def test_builder_rejects_duplicate_question_ids_case_insensitively(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    bundle = qa_root / "issue-0001" / "filtered"
    records = load_jsonl(bundle / "questions.jsonl")
    records[1]["question_id"] = "Q-1"
    write_jsonl(bundle / "questions.jsonl", records)

    with pytest.raises(
        ValueError,
        match="Duplicate QA question_id after case-folding",
    ):
        build_hybrid_dataset(
            multi,
            qa_root,
            task_root,
            tmp_path / "output",
        )


def test_builder_rejects_duplicate_qa_instances_case_insensitively(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    create_qa_bundle(
        qa_root,
        "issue-0002",
        "INSTANCE_REPO__TASK-1",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate QA instance_id after case-folding",
    ):
        build_hybrid_dataset(
            multi,
            qa_root,
            task_root,
            tmp_path / "output",
        )


def test_builder_reports_or_rejects_unused_qa(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    create_qa_bundle(
        qa_root,
        "issue-0003",
        "instance_repo__unknown",
    )

    with pytest.raises(ValueError, match="unknown multi-prompt instances"):
        build_hybrid_dataset(
            multi,
            qa_root,
            task_root,
            tmp_path / "rejected",
        )

    result = build_hybrid_dataset(
        multi,
        qa_root,
        task_root,
        tmp_path / "allowed",
        allow_unused_qa=True,
    )
    assert result["coverage"]["unused_qa_bundles"] == ["instance_repo__unknown"]


def test_builder_rejects_stale_execution_hash(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    bundle = qa_root / "issue-0001" / "filtered"
    report = json.loads((bundle / "golden-proof-executions.json").read_text())
    report["executions"][0]["source_sha256"] = "0" * 64
    write_json(bundle / "golden-proof-executions.json", report)

    with pytest.raises(
        ValueError,
        match="Golden proof execution hash differs",
    ):
        build_hybrid_dataset(
            multi,
            qa_root,
            task_root,
            tmp_path / "output",
        )


def test_builder_rejects_changed_golden_proof_source(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    bundle = qa_root / "issue-0001" / "filtered"
    proof = bundle / "golden-proofs" / "instance_repo__task-1" / "q-1.sh"
    proof.write_text("#!/usr/bin/env bash\nexit 1\n")

    with pytest.raises(
        ValueError,
        match="Golden proof source hash differs",
    ):
        build_hybrid_dataset(
            multi,
            qa_root,
            task_root,
            tmp_path / "output",
        )


def test_builder_records_missing_task_directories(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    (task_root / "instance_repo__task-2").rmdir()

    result = build_hybrid_dataset(
        multi,
        qa_root,
        task_root,
        tmp_path / "output",
    )

    assert result["coverage"]["missing_task_directories"] == ["instance_repo__task-2"]


def test_builder_treats_empty_validated_bundle_as_no_qa(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    bundle = qa_root / "issue-0001" / "filtered"
    for path in sorted(bundle.glob("**/*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    write_jsonl(bundle / "questions.jsonl", [])
    write_json(
        bundle / "metadata.json",
        {
            "schema_version": "2.0",
            "instance_id": "instance_repo__task-1",
        },
    )

    result = build_hybrid_dataset(
        multi,
        qa_root,
        task_root,
        tmp_path / "output",
    )

    assert result["coverage"]["instances_with_qa"] == 0
    assert result["coverage"]["instances_without_qa"] == 2
    first = load_jsonl(tmp_path / "output" / "hybrid-prompts.jsonl")[0]
    assert first["qa_eligible"] is False
    assert all(prompt["kind"] == "issue" for prompt in first["prompts"])


def test_builder_reads_controller_execution_report_from_raw_bundle(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    issue_root = qa_root / "issue-0001"
    filtered_report = issue_root / "filtered" / "golden-proof-executions.json"
    raw_report = issue_root / "raw" / "golden-proof-executions.json"
    raw_report.parent.mkdir()
    filtered_report.replace(raw_report)

    result = build_hybrid_dataset(
        multi,
        qa_root,
        task_root,
        tmp_path / "output",
    )

    assert result["coverage"]["instances_with_qa"] == 1
    bundle_source = result["metadata"]["sources"]["qa_run_root"]["bundles"][0]
    assert set(bundle_source) == {"bundle_dir", "instance_id", "sha256"}
    assert len(bundle_source["sha256"]) == 64


def test_builder_rejects_non_empty_output(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing").write_text("data")

    with pytest.raises(ValueError, match="Output directory is not empty"):
        build_hybrid_dataset(multi, qa_root, task_root, output)


def test_build_qa_multistep_cli(tmp_path):
    multi, qa_root, task_root = create_sources(tmp_path)
    output = tmp_path / "output"

    result = CliRunner().invoke(
        app,
        [
            "build-qa-multistep",
            "--multi-prompt-records",
            str(multi),
            "--qa-run-root",
            str(qa_root),
            "--task-root",
            str(task_root),
            "--output-dir",
            str(output),
            "--task-name",
            "instance_repo__task-1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote 1 hybrid records" in result.output
    records = load_jsonl(output / "hybrid-prompts.jsonl")
    assert [record["instance_id"] for record in records] == ["instance_repo__task-1"]
