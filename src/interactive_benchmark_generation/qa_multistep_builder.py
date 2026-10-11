from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from interactive_benchmark_generation.qa_models import QA_SCHEMA_V2

HYBRID_PROMPT_SCHEMA = "qa_multistep_prompt/v1"
HYBRID_BUILDER_VERSION = "1"
OUTPUT_FILES = (
    "hybrid-prompts.jsonl",
    "metadata.json",
    "coverage.json",
    "validation.json",
)
PRIVATE_FIELDS = {
    "accepted_answers",
    "adversarial_review",
    "answer_contract",
    "assessments",
    "causal_chain",
    "documentation",
    "evidence",
    "evidence_hops",
    "fact",
    "fact_id",
    "golden_answer",
    "golden_proof",
    "golden_proof_validation",
    "incorrect_answers",
    "judge",
    "judge_result",
    "probe_path",
    "probe_validation",
    "proof_source",
    "reasoning",
    "reasoning_chain",
    "review",
    "trajectory",
}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _canonical_jsonl(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json(record) for record in records)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise ValueError(f"Could not read {path}: {error}") from error


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"Could not read {path}: {error}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    return records


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    return value


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _private_paths(value: object, path: str = "$") -> list[str]:
    paths = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key.casefold() in PRIVATE_FIELDS:
                paths.append(child_path)
            paths.extend(_private_paths(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_private_paths(item, f"{path}[{index}]"))
    return paths


def _load_multi_prompt_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        records = _load_jsonl(path)
    else:
        value = _load_json(path)
        if not isinstance(value, list) or not all(isinstance(record, dict) for record in value):
            raise ValueError("Multi-prompt JSON must contain an array of objects")
        records = value
    if not records:
        raise ValueError("Multi-prompt records must not be empty")
    seen = {}
    for index, record in enumerate(records):
        instance_id = _required_string(
            record.get("instance_id"),
            f"multi-prompt record {index} instance_id",
        )
        key = instance_id.casefold()
        if key in seen:
            raise ValueError(
                f"Duplicate multi-prompt instance_id after case-folding: {seen[key]!r} and {instance_id!r}"
            )
        seen[key] = instance_id
        parts = record.get("problem_statements")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"problem_statements must be a non-empty list for {instance_id}")
        for part_index, part in enumerate(parts):
            _required_text(
                part,
                f"problem_statements[{part_index}] for {instance_id}",
            )
        leaked = _private_paths(record)
        if leaked:
            raise ValueError(f"Multi-prompt record {instance_id} contains private fields: {leaked}")
    return records


def _discover_bundle_dirs(qa_run_root: Path) -> list[Path]:
    direct = qa_run_root / "questions.jsonl"
    if direct.is_file():
        return [qa_run_root]
    return sorted(path.parent for path in qa_run_root.glob("**/filtered/questions.jsonl") if path.is_file())


def _execution_report_path(bundle_dir: Path) -> Path:
    for name in ("golden-proof-executions.json", "proof-executions.json"):
        direct = bundle_dir / name
        if direct.is_file():
            return direct
    if bundle_dir.name == "filtered":
        for name in ("golden-proof-executions.json", "proof-executions.json"):
            raw = bundle_dir.parent / "raw" / name
            if raw.is_file():
                return raw
    raise ValueError(f"Missing trusted golden proof execution report for {bundle_dir}")


def _load_golden_executions(path: Path) -> dict[str, dict[str, Any]]:
    report = _load_json(path)
    values = report.get("executions") if isinstance(report, dict) else report
    if not isinstance(values, list):
        raise ValueError(f"Invalid proof execution report: {path}")
    executions = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"Invalid proof execution record: {path}")
        if value.get("protocol") != "qa_proof_execution/v1":
            raise ValueError(f"Invalid proof execution protocol: {path}")
        if value.get("proof_kind") != "golden":
            continue
        question_id = _required_string(
            value.get("question_id"),
            f"proof execution question_id in {path}",
        )
        key = question_id.casefold()
        if key in executions:
            raise ValueError(f"Duplicate golden proof execution for {question_id}: {path}")
        source_sha256 = _required_string(
            value.get("source_sha256"),
            f"proof execution source_sha256 for {question_id}",
        )
        required_true = (
            "submitted",
            "executed",
            "success",
            "repository_state_unchanged",
            "repository_dependent",
            "deterministic",
        )
        if any(value.get(field) is not True for field in required_true):
            raise ValueError(f"Golden proof execution is not trusted for {question_id}")
        if "evidence_valid" in value and value["evidence_valid"] is not True:
            raise ValueError(f"Golden proof execution evidence is invalid for {question_id}")
        executions[key] = {
            "question_id": question_id,
            "source_sha256": source_sha256,
        }
    return executions


def _metadata_instance_id(bundle_dir: Path) -> str | None:
    path = bundle_dir / "metadata.json"
    if not path.is_file():
        return None
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"QA metadata must contain an object: {path}")
    instance_id = value.get("instance_id")
    if instance_id is None:
        return None
    return _required_string(instance_id, f"instance_id in {path}")


def _relative_input_files(
    input_files: dict[Path, str],
    qa_run_root: Path,
) -> list[dict[str, str]]:
    root = qa_run_root.resolve()
    return [
        {
            "path": path.resolve().relative_to(root).as_posix(),
            "sha256": digest,
        }
        for path, digest in sorted(
            input_files.items(),
            key=lambda item: item[0].resolve().relative_to(root).as_posix(),
        )
    ]


def _load_qa_bundle(
    bundle_dir: Path,
    qa_run_root: Path,
) -> dict[str, Any]:
    questions_path = bundle_dir / "questions.jsonl"
    records = _load_jsonl(questions_path)
    metadata_path = bundle_dir / "metadata.json"
    metadata_instance_id = _metadata_instance_id(bundle_dir)
    if not records:
        if metadata_instance_id is None:
            raise ValueError(f"Empty QA bundle requires metadata instance_id: {bundle_dir}")
        input_files = {
            questions_path: _sha256_file(questions_path),
            metadata_path: _sha256_file(metadata_path),
        }
        digest_payload = _relative_input_files(input_files, qa_run_root)
        return {
            "instance_id": metadata_instance_id,
            "bundle_dir": bundle_dir.resolve().relative_to(qa_run_root.resolve()).as_posix(),
            "questions": [],
            "input_files": digest_payload,
            "sha256": _sha256_bytes(_canonical_json(digest_payload)),
        }
    instance_ids = [_required_string(record.get("instance_id"), "QA instance_id") for record in records]
    if len({instance_id.casefold() for instance_id in instance_ids}) != 1:
        raise ValueError(f"QA bundle contains multiple instance IDs: {bundle_dir}")
    instance_id = instance_ids[0]
    if any(value != instance_id for value in instance_ids):
        raise ValueError(f"QA bundle has case-only instance ID collisions: {bundle_dir}")
    if metadata_instance_id is not None and metadata_instance_id.casefold() != instance_id.casefold():
        raise ValueError(f"QA metadata instance_id differs from questions: {bundle_dir}")
    report_path = _execution_report_path(bundle_dir)
    executions = _load_golden_executions(report_path)
    questions = []
    question_ids = {}
    input_files = {
        questions_path: _sha256_file(questions_path),
        report_path: _sha256_file(report_path),
    }
    if metadata_path.is_file():
        input_files[metadata_path] = _sha256_file(metadata_path)
    for record in records:
        if record.get("schema_version") != QA_SCHEMA_V2:
            raise ValueError(f"QA question must use schema {QA_SCHEMA_V2}: {bundle_dir}")
        question_id = _required_string(
            record.get("question_id"),
            f"question_id in {bundle_dir}",
        )
        question_key = question_id.casefold()
        if question_key in question_ids:
            raise ValueError(
                "Duplicate QA question_id after case-folding for "
                f"{instance_id}: {question_ids[question_key]!r} and "
                f"{question_id!r}"
            )
        question_ids[question_key] = question_id
        question = _required_text(
            record.get("question"),
            f"question {question_id}",
        )
        proof = record.get("golden_proof")
        if not isinstance(proof, dict):
            raise ValueError(f"Missing golden_proof for {question_id}")
        proof_sha256 = _required_string(
            proof.get("sha256"),
            f"golden_proof.sha256 for {question_id}",
        )
        proof_path_value = _required_string(
            proof.get("path"),
            f"golden_proof.path for {question_id}",
        )
        proof_path = bundle_dir / proof_path_value
        if not proof_path.is_file():
            raise ValueError(f"Missing golden proof source: {proof_path}")
        actual_proof_sha256 = _sha256_file(proof_path)
        if actual_proof_sha256 != proof_sha256:
            raise ValueError(f"Golden proof source hash differs for {question_id}")
        input_files[proof_path] = actual_proof_sha256
        validation = record.get("golden_proof_validation")
        # The four trusted fields must all hold; the bundle may carry more. Dict equality was the
        # original check and it broke the moment GoldenProofValidation gained patch_stable, since
        # a bundle rewritten by the current model serializes the new field while every bundle
        # written before it does not. filter_validated_bundle checks these four individually.
        required_validation = (
            "operationally_valid",
            "repository_dependent",
            "deterministic",
            "repository_state_unchanged",
        )
        if not isinstance(validation, dict) or any(
            validation.get(field) is not True for field in required_validation
        ):
            raise ValueError(f"Question lacks trusted golden proof validation: {question_id}")
        review = record.get("adversarial_review")
        if not isinstance(review, dict) or review.get("accepted") is not True:
            raise ValueError(f"Question lacks an accepted adversarial review: {question_id}")
        execution = executions.get(question_id.casefold())
        if execution is None:
            raise ValueError(f"Missing trusted golden proof execution for {question_id}")
        if execution["source_sha256"] != proof_sha256:
            raise ValueError(f"Golden proof execution hash differs for {question_id}")
        questions.append({"question_id": question_id, "text": question})
    digest_payload = _relative_input_files(input_files, qa_run_root)
    return {
        "instance_id": instance_id,
        "bundle_dir": bundle_dir.resolve().relative_to(qa_run_root.resolve()).as_posix(),
        "questions": questions,
        "input_files": digest_payload,
        "sha256": _sha256_bytes(_canonical_json(digest_payload)),
    }


def _load_qa_bundles(qa_run_root: Path) -> dict[str, dict[str, Any]]:
    if not qa_run_root.is_dir():
        raise ValueError(f"QA run root is not a directory: {qa_run_root}")
    bundle_dirs = _discover_bundle_dirs(qa_run_root)
    bundles = {}
    exact_ids = {}
    for bundle_dir in bundle_dirs:
        bundle = _load_qa_bundle(bundle_dir, qa_run_root)
        instance_id = bundle["instance_id"]
        key = instance_id.casefold()
        if key in bundles:
            raise ValueError(f"Duplicate QA instance_id after case-folding: {exact_ids[key]!r} and {instance_id!r}")
        bundles[key] = bundle
        exact_ids[key] = instance_id
    return bundles


def _select_records(
    records: list[dict[str, Any]],
    task_names: list[str] | None,
) -> list[dict[str, Any]]:
    if not task_names:
        return records
    requested = {}
    for task_name in task_names:
        name = _required_string(task_name, "task_name")
        key = name.casefold()
        if key in requested:
            raise ValueError(f"Duplicate selected task name: {name}")
        requested[key] = name
    available = {record["instance_id"].casefold(): record["instance_id"] for record in records}
    missing = sorted(name for key, name in requested.items() if key not in available)
    if missing:
        raise ValueError(f"Selected tasks are absent from multi-prompt input: {missing}")
    return [record for record in records if record["instance_id"].casefold() in requested]


def _missing_task_directories(
    records: list[dict[str, Any]],
    task_root: Path,
) -> list[str]:
    if not task_root.is_dir():
        raise ValueError(f"Task root is not a directory: {task_root}")
    directory_names = [path.name for path in task_root.iterdir() if path.is_dir()]
    folded = {}
    for name in directory_names:
        key = name.casefold()
        if key in folded:
            raise ValueError(f"Task root has case-only directory collisions: {folded[key]!r} and {name!r}")
        folded[key] = name
    missing = []
    for record in records:
        instance_id = record["instance_id"]
        actual_name = folded.get(instance_id.casefold())
        if actual_name is None:
            missing.append(instance_id)
        elif actual_name != instance_id:
            raise ValueError(f"Task directory casing differs for {instance_id}: {actual_name}")
    return missing


def _validate_public_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "instance_id",
        "qa_eligible",
        "prompts",
        "source_digests",
        "record_sha256",
    }
    if set(record) != required:
        raise ValueError(f"Hybrid record fields differ: {sorted(set(record) - required)}")
    if record["schema_version"] != HYBRID_PROMPT_SCHEMA:
        raise ValueError("Hybrid record has an invalid schema_version")
    _required_string(record["instance_id"], "instance_id")
    if not isinstance(record["qa_eligible"], bool):
        raise ValueError("qa_eligible must be a boolean")
    source_digests = record["source_digests"]
    if not isinstance(source_digests, dict) or set(source_digests) != {
        "multi_prompt_record_sha256",
        "qa_bundle_sha256",
    }:
        raise ValueError("source_digests fields are invalid")
    for name, digest in source_digests.items():
        if digest is None and name == "qa_bundle_sha256":
            continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    prompts = record["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Hybrid prompts must be a non-empty list")
    issue_started = False
    issue_indices = []
    question_ids = set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("Hybrid prompts must be objects")
        kind = prompt.get("kind")
        if kind == "qa":
            if set(prompt) != {"kind", "question_id", "text"}:
                raise ValueError("QA prompt fields are invalid")
            if issue_started:
                raise ValueError("QA prompts must precede issue prompts")
            question_id = _required_string(prompt["question_id"], "question_id")
            if question_id.casefold() in question_ids:
                raise ValueError(f"Duplicate question_id: {question_id}")
            question_ids.add(question_id.casefold())
        elif kind == "issue":
            if set(prompt) != {"kind", "part_index", "text"}:
                raise ValueError("Issue prompt fields are invalid")
            issue_started = True
            part_index = prompt["part_index"]
            if not isinstance(part_index, int) or isinstance(part_index, bool):
                raise ValueError("Issue part_index must be an integer")
            issue_indices.append(part_index)
        else:
            raise ValueError(f"Unknown hybrid prompt kind: {kind!r}")
        _required_text(prompt["text"], "prompt text")
    if issue_indices and issue_indices != list(range(len(issue_indices))):
        raise ValueError("Issue part_index values must be contiguous from zero")
    if record["qa_eligible"] != bool(question_ids):
        raise ValueError("qa_eligible must match QA prompt availability")
    leaked = _private_paths(record)
    if leaked:
        raise ValueError(f"Public hybrid record contains private fields: {leaked}")
    digest_input = dict(record)
    digest = digest_input.pop("record_sha256")
    if digest != _sha256_bytes(_canonical_json(digest_input)):
        raise ValueError("Hybrid record SHA-256 is invalid")


def _build_artifacts(
    multi_prompt_records: Path,
    qa_run_root: Path,
    task_root: Path,
    task_names: list[str] | None,
    allow_unused_qa: bool,
) -> dict[str, bytes]:
    source_records = _load_multi_prompt_records(multi_prompt_records)
    selected_records = _select_records(source_records, task_names)
    qa_bundles = _load_qa_bundles(qa_run_root)
    all_multi_keys = {record["instance_id"].casefold() for record in source_records}
    unused_bundles = sorted(bundle["instance_id"] for key, bundle in qa_bundles.items() if key not in all_multi_keys)
    if unused_bundles and not allow_unused_qa:
        raise ValueError(f"QA bundles refer to unknown multi-prompt instances: {unused_bundles}")
    missing_task_directories = _missing_task_directories(
        selected_records,
        task_root,
    )
    hybrid_records = []
    qa_source_entries = []
    question_counts = []
    for source_record in selected_records:
        instance_id = source_record["instance_id"]
        bundle = qa_bundles.get(instance_id.casefold())
        questions = bundle["questions"] if bundle is not None else []
        prompts = [
            {
                "kind": "qa",
                "question_id": question["question_id"],
                "text": question["text"],
            }
            for question in questions
        ]
        prompts.extend(
            {
                "kind": "issue",
                "part_index": index,
                "text": text,
            }
            for index, text in enumerate(source_record["problem_statements"])
        )
        source_digests = {
            "multi_prompt_record_sha256": _sha256_bytes(_canonical_json(source_record)),
            "qa_bundle_sha256": bundle["sha256"] if bundle is not None else None,
        }
        record_without_digest = {
            "schema_version": HYBRID_PROMPT_SCHEMA,
            "instance_id": instance_id,
            "qa_eligible": bool(questions),
            "prompts": prompts,
            "source_digests": source_digests,
        }
        hybrid_record = {
            **record_without_digest,
            "record_sha256": _sha256_bytes(_canonical_json(record_without_digest)),
        }
        _validate_public_record(hybrid_record)
        hybrid_records.append(hybrid_record)
        question_counts.append(len(questions))
        if bundle is not None:
            qa_source_entries.append(
                {
                    "instance_id": bundle["instance_id"],
                    "bundle_dir": bundle["bundle_dir"],
                    "sha256": bundle["sha256"],
                }
            )
    hybrid_bytes = _canonical_jsonl(hybrid_records)
    coverage = {
        "schema_version": HYBRID_PROMPT_SCHEMA,
        "multi_step_instances": len(hybrid_records),
        "instances_with_qa": sum(count > 0 for count in question_counts),
        "instances_without_qa": sum(count == 0 for count in question_counts),
        "questions": sum(question_counts),
        "question_count_distribution": {
            str(count): frequency for count, frequency in sorted(Counter(question_counts).items())
        },
        "unused_qa_bundles": unused_bundles,
        "duplicate_records": [],
        "rejected_records": [],
        "missing_task_directories": missing_task_directories,
    }
    coverage_bytes = _canonical_json(coverage)
    metadata = {
        "schema_version": HYBRID_PROMPT_SCHEMA,
        "builder_version": HYBRID_BUILDER_VERSION,
        "sources": {
            "multi_prompt_records": {
                "name": multi_prompt_records.name,
                "sha256": _sha256_file(multi_prompt_records),
            },
            "qa_run_root": {
                "name": qa_run_root.name,
                "bundles": qa_source_entries,
            },
            "task_root": {
                "name": task_root.name,
            },
        },
        "creation_options": {
            "allow_unused_qa": allow_unused_qa,
            "task_names": list(task_names or []),
        },
        "output": {
            "path": "hybrid-prompts.jsonl",
            "sha256": _sha256_bytes(hybrid_bytes),
            "records": len(hybrid_records),
        },
    }
    metadata_bytes = _canonical_json(metadata)
    validation = {
        "schema_version": HYBRID_PROMPT_SCHEMA,
        "valid": True,
        "checks": {
            "casefold_instance_ids_unique": True,
            "casefold_question_ids_unique_per_instance": True,
            "golden_proof_sources_match": True,
            "trusted_golden_provenance_match": True,
            "qa_first_ordering": True,
            "issue_parts_contiguous": True,
            "private_fields_absent": True,
            "tasks_without_qa_retained": True,
        },
        "artifact_sha256": {
            "hybrid-prompts.jsonl": _sha256_bytes(hybrid_bytes),
            "metadata.json": _sha256_bytes(metadata_bytes),
            "coverage.json": _sha256_bytes(coverage_bytes),
        },
        "records_validated": len(hybrid_records),
        "questions_validated": sum(question_counts),
    }
    validation_bytes = _canonical_json(validation)
    return {
        "hybrid-prompts.jsonl": hybrid_bytes,
        "metadata.json": metadata_bytes,
        "coverage.json": coverage_bytes,
        "validation.json": validation_bytes,
    }


def _verify_artifacts(output_dir: Path, artifacts: dict[str, bytes]) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"Rebuild verification requires an existing output directory: {output_dir}")
    actual_names = sorted(path.name for path in output_dir.iterdir())
    if actual_names != sorted(OUTPUT_FILES):
        raise ValueError(f"Rebuild output files differ: expected={sorted(OUTPUT_FILES)}, actual={actual_names}")
    for name, expected in artifacts.items():
        actual = (output_dir / name).read_bytes()
        if actual != expected:
            raise ValueError(f"Rebuild differs byte-for-byte: {name}")


def _write_artifacts_atomic(
    output_dir: Path,
    artifacts: dict[str, bytes],
) -> None:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {output_dir}")
        output_dir.rmdir()
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    try:
        for name, value in artifacts.items():
            (temporary_dir / name).write_bytes(value)
        temporary_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def build_hybrid_dataset(
    multi_prompt_records: Path,
    qa_run_root: Path,
    task_root: Path,
    output_dir: Path,
    task_names: list[str] | None = None,
    allow_unused_qa: bool = False,
    verify_rebuild: bool = False,
) -> dict[str, Any]:
    artifacts = _build_artifacts(
        multi_prompt_records=multi_prompt_records,
        qa_run_root=qa_run_root,
        task_root=task_root,
        task_names=task_names,
        allow_unused_qa=allow_unused_qa,
    )
    if verify_rebuild:
        _verify_artifacts(output_dir, artifacts)
    else:
        _write_artifacts_atomic(output_dir, artifacts)
    return {
        "output_dir": str(output_dir.resolve()),
        "verified_rebuild": verify_rebuild,
        "metadata": json.loads(artifacts["metadata.json"]),
        "coverage": json.loads(artifacts["coverage.json"]),
        "validation": json.loads(artifacts["validation.json"]),
    }
