from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from interactive_benchmark_generation.dataset_io import write_json
from interactive_benchmark_generation.qa_io import (
    load_generation_bundle,
    prune_unusable_questions,
    replace_questions,
    validate_golden_proofs,
    validate_question_complexity,
)
from interactive_benchmark_generation.qa_models import (
    QA_SCHEMA_V2,
    GenerationBundle,
    QARecord,
)

QA_QUESTION_REPAIR_REQUEST_PROTOCOL = "qa_question_repair_request/v1"
QA_QUESTION_REGENERATION_REQUEST_PROTOCOL = "qa_question_regeneration_request/v1"
QA_QUESTION_REPLACEMENTS_PROTOCOL = "qa_question_replacements/v1"
QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL = "qa_dual_state_diagnostics/v1"
QA_QUESTION_REPLACEMENT_HISTORY_PROTOCOL = "qa_question_replacement_history/v1"
_REQUEST_PROTOCOLS = {
    "repair": QA_QUESTION_REPAIR_REQUEST_PROTOCOL,
    "regeneration": QA_QUESTION_REGENERATION_REQUEST_PROTOCOL,
}
_IDENTITY_FIELDS = (
    "instance_id",
    "schema_version",
    "evaluation_mode",
    "question_id",
)
_REPLACEMENT_REQUIRED_FIELDS = {
    "instance_id",
    "schema_version",
    "evaluation_mode",
    "question_id",
    "fact_id",
    "question",
    "category",
    "golden_answer",
    "evidence_hops",
    "reasoning_chain",
    "golden_proof",
    "golden_proof_script",
}
_REPLACEMENT_OPTIONAL_FIELDS = {"flow_scope"}
_DIAGNOSTIC_OUTPUT_LIMIT_BYTES = 65536
_PATCH_CONTENT_MARKERS = (
    "diff --git ",
    "--- a/",
    "+++ b/",
    "@@ ",
)


def _read_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {name} JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _strict_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _question_ids(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def _diagnostic_output(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if len(value.encode("utf-8")) > _DIAGNOSTIC_OUTPUT_LIMIT_BYTES:
        raise ValueError(f"{name} exceeds {_DIAGNOSTIC_OUTPUT_LIMIT_BYTES} UTF-8 bytes")
    if any(marker in value for marker in _PATCH_CONTENT_MARKERS):
        raise ValueError(f"{name} contains private patch content")
    return value


def _failure_codes(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(code, str) or not code.strip() for code in value
    ):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return list(value)


def _patch_application_diagnostic(
    value: object,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    _strict_keys(
        value,
        {
            "exit_code",
            "output_truncated",
            "stdout",
            "stderr",
            "failure_codes",
        },
        set(),
        name,
    )
    exit_code = value["exit_code"]
    if exit_code is not None and (
        not isinstance(exit_code, int) or isinstance(exit_code, bool)
    ):
        raise ValueError(f"{name}.exit_code must be an integer or null")
    if not isinstance(value["output_truncated"], bool):
        raise ValueError(f"{name}.output_truncated must be a boolean")
    return {
        "exit_code": exit_code,
        "output_truncated": value["output_truncated"],
        "stdout": _diagnostic_output(value["stdout"], f"{name}.stdout"),
        "stderr": _diagnostic_output(value["stderr"], f"{name}.stderr"),
        "failure_codes": _failure_codes(
            value["failure_codes"],
            f"{name}.failure_codes",
        ),
    }


def _state_diagnostic(
    value: object,
    name: str,
    *,
    post_patch: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    required = {
        "passed",
        "exit_code",
        "timed_out",
        "output_truncated",
        "failure_codes",
    }
    optional = {
        "repository_state_unchanged",
        "stdout",
        "stderr",
    }
    if post_patch:
        optional.update({"applied", "patch_application"})
    _strict_keys(value, required, optional, name)
    if not isinstance(value["passed"], bool):
        raise ValueError(f"{name}.passed must be a boolean")
    if value["exit_code"] is not None and (
        not isinstance(value["exit_code"], int) or isinstance(value["exit_code"], bool)
    ):
        raise ValueError(f"{name}.exit_code must be an integer or null")
    for field in ("timed_out", "output_truncated"):
        if not isinstance(value[field], bool):
            raise ValueError(f"{name}.{field} must be a boolean")
    codes = _failure_codes(value["failure_codes"], f"{name}.failure_codes")
    result = {
        "passed": value["passed"],
        "exit_code": value["exit_code"],
        "timed_out": value["timed_out"],
        "output_truncated": value["output_truncated"],
        "failure_codes": list(codes),
    }
    for field in (
        "repository_state_unchanged",
        "applied",
    ):
        if field not in value:
            continue
        if value[field] is not None and not isinstance(value[field], bool):
            raise ValueError(f"{name}.{field} must be a boolean or null")
        result[field] = value[field]
    if "patch_application" in value:
        result["patch_application"] = _patch_application_diagnostic(
            value["patch_application"],
            f"{name}.patch_application",
        )
    for field in ("stdout", "stderr"):
        if field in value:
            result[field] = _diagnostic_output(
                value[field],
                f"{name}.{field}",
            )
    return result


def load_dual_state_diagnostics(
    path: Path,
    *,
    instance_id: str,
    expected_question_ids: set[str],
) -> dict[str, dict[str, Any]]:
    value = _read_object(path, "dual-state diagnostics")
    _strict_keys(
        value,
        {"protocol", "instance_id", "questions"},
        set(),
        "dual-state diagnostics",
    )
    if value["protocol"] != QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL:
        raise ValueError("Dual-state diagnostics protocol is invalid")
    if value["instance_id"] != instance_id:
        raise ValueError("Dual-state diagnostics instance_id differs")
    questions = value["questions"]
    if not isinstance(questions, list):
        raise ValueError("Dual-state diagnostics questions must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(questions):
        if not isinstance(item, dict):
            raise ValueError(f"Dual-state diagnostic {index} must be an object")
        _strict_keys(
            item,
            {"question_id", "pristine", "post_patch"},
            set(),
            f"dual-state diagnostic {index}",
        )
        question_id = item["question_id"]
        if question_id not in expected_question_ids:
            raise ValueError(f"Unknown dual-state diagnostic question_id: {question_id}")
        if question_id in by_id:
            raise ValueError(f"Duplicate dual-state diagnostic question_id: {question_id}")
        by_id[question_id] = {
            "pristine": _state_diagnostic(
                item["pristine"],
                f"{question_id}.pristine",
                post_patch=False,
            ),
            "post_patch": _state_diagnostic(
                item["post_patch"],
                f"{question_id}.post_patch",
                post_patch=True,
            ),
        }
    return by_id


def build_question_replacement_request(
    bundle_dir: Path,
    output_path: Path,
    *,
    mode: str,
    question_ids: list[str],
    attempt: int,
    max_attempts: int,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in _REQUEST_PROTOCOLS:
        raise ValueError("Replacement mode must be repair or regeneration")
    attempt = _positive_integer(attempt, "attempt")
    max_attempts = _positive_integer(max_attempts, "max_attempts")
    if attempt > max_attempts:
        raise ValueError("attempt must not exceed max_attempts")
    requested_ids = _question_ids(question_ids, "question_ids")
    bundle = load_generation_bundle(bundle_dir)
    if bundle.schema_version != QA_SCHEMA_V2:
        raise ValueError("Question replacement requires a v2 QA bundle")
    records = {record.question_id: record for record in bundle.questions}
    unknown = set(requested_ids) - set(records)
    if unknown:
        raise ValueError(f"Unknown replacement question IDs: {sorted(unknown)}")
    diagnostics = {}
    if diagnostics_path is not None:
        diagnostics = load_dual_state_diagnostics(
            diagnostics_path,
            instance_id=bundle.instance_id,
            expected_question_ids=set(requested_ids),
        )
        missing = set(requested_ids) - set(diagnostics)
        if missing:
            raise ValueError(f"Dual-state diagnostics are missing IDs: {sorted(missing)}")
    questions = []
    for question_id in requested_ids:
        record = records[question_id]
        value = record.to_dict()
        value["golden_proof_script"] = bundle.golden_proofs[record.golden_proof.path]
        questions.append(value)
    request = {
        "protocol": _REQUEST_PROTOCOLS[mode],
        "instance_id": bundle.instance_id,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "immutable_fields": list(_IDENTITY_FIELDS),
        "question_ids": requested_ids,
        "questions": questions,
        "diagnostics": diagnostics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, request)
    return request


def _validate_request(
    value: dict[str, Any],
    bundle: GenerationBundle,
) -> tuple[list[str], int, int]:
    _strict_keys(
        value,
        {
            "protocol",
            "instance_id",
            "attempt",
            "max_attempts",
            "immutable_fields",
            "question_ids",
            "questions",
            "diagnostics",
        },
        set(),
        "replacement request",
    )
    if value["protocol"] not in _REQUEST_PROTOCOLS.values():
        raise ValueError("Replacement request protocol is invalid")
    if value["instance_id"] != bundle.instance_id:
        raise ValueError("Replacement request instance_id differs")
    if value["immutable_fields"] != list(_IDENTITY_FIELDS):
        raise ValueError("Replacement request immutable_fields differ")
    attempt = _positive_integer(value["attempt"], "attempt")
    max_attempts = _positive_integer(value["max_attempts"], "max_attempts")
    if attempt > max_attempts:
        raise ValueError("attempt must not exceed max_attempts")
    requested_ids = _question_ids(
        value["question_ids"],
        "replacement request question_ids",
    )
    if set(requested_ids) - {record.question_id for record in bundle.questions}:
        raise ValueError("Replacement request contains unknown question IDs")
    if not isinstance(value["questions"], list):
        raise ValueError("Replacement request questions must be a list")
    request_record_ids = [item.get("question_id") for item in value["questions"] if isinstance(item, dict)]
    if request_record_ids != requested_ids:
        raise ValueError("Replacement request question records differ")
    diagnostics = value["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise ValueError("Replacement request diagnostics must be an object")
    if set(diagnostics) - set(requested_ids):
        raise ValueError("Replacement request diagnostics contain unknown IDs")
    for question_id, item in diagnostics.items():
        if not isinstance(item, dict):
            raise ValueError(f"Replacement request diagnostic is invalid: {question_id}")
        _strict_keys(
            item,
            {"pristine", "post_patch"},
            set(),
            f"replacement request diagnostic {question_id}",
        )
        _state_diagnostic(
            item["pristine"],
            f"{question_id}.pristine",
            post_patch=False,
        )
        _state_diagnostic(
            item["post_patch"],
            f"{question_id}.post_patch",
            post_patch=True,
        )
    return requested_ids, attempt, max_attempts


def _replacement_record(
    replacement: dict[str, Any],
    original: QARecord,
    bundle: GenerationBundle,
) -> tuple[QARecord, str]:
    _strict_keys(
        replacement,
        _REPLACEMENT_REQUIRED_FIELDS,
        _REPLACEMENT_OPTIONAL_FIELDS,
        f"replacement for {original.question_id}",
    )
    for field in _IDENTITY_FIELDS:
        expected = getattr(original, field)
        actual = replacement[field]
        if hasattr(expected, "value"):
            expected = expected.value
        if actual != expected:
            raise ValueError(f"Replacement changed immutable field {field} for {original.question_id}")
    script = replacement["golden_proof_script"]
    if not isinstance(script, str) or not script.strip():
        raise ValueError(f"Replacement golden_proof_script is empty for {original.question_id}")
    digest = hashlib.sha256(script.encode()).hexdigest()
    proof = replacement["golden_proof"]
    if not isinstance(proof, dict) or proof.get("sha256") != digest:
        raise ValueError(f"Replacement golden proof SHA-256 mismatch for {original.question_id}")
    facts = {fact.fact_id: fact for fact in bundle.facts}
    fact = facts.get(replacement["fact_id"])
    if fact is None:
        raise ValueError(f"Replacement uses unknown fact_id for {original.question_id}: {replacement['fact_id']}")
    record_value = {key: value for key, value in replacement.items() if key != "golden_proof_script"}
    record_value["fact"] = fact.fact
    record = QARecord.from_dict(record_value)
    candidate = GenerationBundle(
        instance_id=bundle.instance_id,
        documentation=bundle.documentation,
        facts=(fact,),
        questions=(record,),
        probes={},
        golden_proofs={record.golden_proof.path: script},
        metadata={},
        debug={},
        schema_version=bundle.schema_version,
    )
    validate_question_complexity(candidate)
    validate_golden_proofs(candidate)
    return record, script


def _bundle_value(
    bundle: GenerationBundle,
    records: list[QARecord],
    scripts: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": bundle.schema_version,
        "instance_id": bundle.instance_id,
        "documentation": bundle.documentation,
        "facts": [fact.to_dict() for fact in bundle.facts],
        "questions": [record.to_dict() for record in records],
        "probes": {},
        "golden_proofs": scripts,
        "metadata": dict(bundle.metadata),
        "debug": dict(bundle.debug),
    }


def _atomic_write_proof(path: Path, script: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(script, encoding="utf-8")
        try:
            temporary.chmod(0o755)
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
        try:
            os.replace(temporary, path)
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
            path.write_text(script, encoding="utf-8")
            try:
                path.chmod(0o755)
            except PermissionError as chmod_error:
                if chmod_error.errno != errno.EPERM:
                    raise
            if path.read_text(encoding="utf-8") != script:
                raise OSError(
                    f"Failed to persist golden proof: {path}"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _append_history(
    bundle_dir: Path,
    entry: dict[str, Any],
) -> None:
    path = bundle_dir / "question-replacement-attempts.json"
    if path.exists():
        history = _read_object(path, "replacement history")
        _strict_keys(
            history,
            {"protocol", "attempts"},
            set(),
            "replacement history",
        )
        if history["protocol"] != QA_QUESTION_REPLACEMENT_HISTORY_PROTOCOL:
            raise ValueError("Replacement history protocol is invalid")
        attempts = history["attempts"]
        if not isinstance(attempts, list):
            raise ValueError("Replacement history attempts must be a list")
    else:
        attempts = []
    write_json(
        path,
        {
            "protocol": QA_QUESTION_REPLACEMENT_HISTORY_PROTOCOL,
            "attempts": [*attempts, entry],
        },
    )


def apply_question_replacements(
    bundle_dir: Path,
    request_path: Path,
    replacements_path: Path,
) -> dict[str, Any]:
    bundle = load_generation_bundle(bundle_dir)
    if bundle.schema_version != QA_SCHEMA_V2:
        raise ValueError("Question replacement requires a v2 QA bundle")
    request = _read_object(request_path, "replacement request")
    requested_ids, attempt, max_attempts = _validate_request(request, bundle)
    response = _read_object(replacements_path, "question replacements")
    _strict_keys(
        response,
        {
            "protocol",
            "request_protocol",
            "instance_id",
            "attempt",
            "replacements",
        },
        set(),
        "question replacements",
    )
    if response["protocol"] != QA_QUESTION_REPLACEMENTS_PROTOCOL:
        raise ValueError("Question replacements protocol is invalid")
    if response["request_protocol"] != request["protocol"]:
        raise ValueError("Question replacements request_protocol differs")
    if response["instance_id"] != bundle.instance_id:
        raise ValueError("Question replacements instance_id differs")
    if response["attempt"] != attempt:
        raise ValueError("Question replacements attempt differs")
    replacements = response["replacements"]
    if not isinstance(replacements, list):
        raise ValueError("Question replacements must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, dict):
            raise ValueError(f"Question replacement {index} must be an object")
        question_id = replacement.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Question replacement {index} has a malformed ID")
        if question_id not in requested_ids:
            raise ValueError(f"Unknown replacement question_id: {question_id}")
        if question_id in by_id:
            raise ValueError(f"Duplicate replacement question_id: {question_id}")
        by_id[question_id] = replacement
    originals = {record.question_id: record for record in bundle.questions}
    candidate_records: dict[str, QARecord] = {}
    candidate_scripts: dict[str, str] = {}
    results = []
    exhausted = attempt == max_attempts
    for question_id in requested_ids:
        replacement = by_id.get(question_id)
        if replacement is None:
            results.append(
                {
                    "question_id": question_id,
                    "status": ("dropped_exhausted" if exhausted else "retained_for_retry"),
                    "error": "replacement is missing",
                }
            )
            continue
        try:
            record, script = _replacement_record(
                replacement,
                originals[question_id],
                bundle,
            )
        except (KeyError, TypeError, ValueError) as error:
            results.append(
                {
                    "question_id": question_id,
                    "status": ("dropped_exhausted" if exhausted else "retained_for_retry"),
                    "error": str(error),
                }
            )
            continue
        candidate_records[question_id] = record
        candidate_scripts[question_id] = script
        results.append(
            {
                "question_id": question_id,
                "status": "candidate",
                "previous_sha256": originals[question_id].golden_proof.sha256,
                "replacement_sha256": record.golden_proof.sha256,
            }
        )
    records = []
    scripts = dict(bundle.golden_proofs)
    for record in bundle.questions:
        question_id = record.question_id
        if question_id in candidate_records:
            candidate = candidate_records[question_id]
            records.append(candidate)
            scripts[candidate.golden_proof.path] = candidate_scripts[question_id]
            continue
        result = next(
            (item for item in results if item["question_id"] == question_id),
            None,
        )
        if result is not None and result["status"] == "dropped_exhausted":
            continue
        records.append(record)
    value, collision_drops = prune_unusable_questions(_bundle_value(bundle, records, scripts))
    final_bundle = GenerationBundle.from_dict(value)
    validate_question_complexity(final_bundle)
    validate_golden_proofs(final_bundle)
    retained_ids = {record.question_id for record in final_bundle.questions}
    for result in results:
        if result["status"] != "candidate":
            continue
        if result["question_id"] in retained_ids:
            result["status"] = "applied"
        else:
            result["status"] = "dropped_collision"
    for record in final_bundle.questions:
        if record.question_id not in candidate_scripts:
            continue
        _atomic_write_proof(
            bundle_dir / record.golden_proof.path,
            candidate_scripts[record.question_id],
        )
    replace_questions(bundle_dir, list(final_bundle.questions))
    metadata_path = bundle_dir / "metadata.json"
    metadata = _read_object(metadata_path, "bundle metadata")
    metadata["n_questions"] = len(final_bundle.questions)
    write_json(metadata_path, metadata)
    summary = {
        "protocol": QA_QUESTION_REPLACEMENT_HISTORY_PROTOCOL,
        "request_protocol": request["protocol"],
        "request": str(request_path),
        "response": str(replacements_path),
        "attempt": attempt,
        "max_attempts": max_attempts,
        "diagnostics": request["diagnostics"],
        "results": results,
        "collision_drops": collision_drops,
        "n_applied": sum(item["status"] == "applied" for item in results),
        "n_retained_for_retry": sum(item["status"] == "retained_for_retry" for item in results),
        "n_dropped": sum(item["status"] in {"dropped_exhausted", "dropped_collision"} for item in results),
        "remaining_question_ids": [record.question_id for record in final_bundle.questions],
    }
    _append_history(bundle_dir, summary)
    return summary
