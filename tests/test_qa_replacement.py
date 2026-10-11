from __future__ import annotations

import errno
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import interactive_benchmark_generation.qa_replacement as qa_replacement
from interactive_benchmark_generation.qa_io import (
    load_generation_bundle,
    load_qa_records,
    write_qa_bundle,
)
from interactive_benchmark_generation.qa_models import (
    GenerationBundle,
    GoldenProofValidation,
    ProofExecutionRecord,
)
from interactive_benchmark_generation.qa_replacement import (
    QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL,
    QA_QUESTION_REPLACEMENTS_PROTOCOL,
    apply_question_replacements,
    build_question_replacement_request,
)
from interactive_benchmark_generation.qa_validation import (
    golden_proof_validation_passes,
)

INSTANCE_ID = "repo__issue"


def proof_script(observation: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'REPO_ROOT="${REPO_ROOT:?required}"\n'
        "PROOF_REQUIRED_TOOLS=(bash)\n"
        "command -v bash >/dev/null || { "
        "printf 'missing required proof tool\\n' >&2; exit 2; }\n"
        '[[ -f "$REPO_ROOT/pyproject.toml" ]] || exit 2\n'
        f"printf '%s\\n' '{observation}'\n"
    )


def hops(count: int = 3) -> list[dict[str, str]]:
    return [
        {
            "evidence_id": f"e{index}",
            "file": "entry.py" if index == 0 else "flow.py",
            "symbol": f"symbol_{index}",
            "claim": f"Evidence claim {index}.",
        }
        for index in range(count)
    ]


def reasoning(count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "step_id": f"s{index}",
            "claim": f"Reasoning claim {index}.",
            "evidence_ids": [f"e{index}"],
        }
        for index in range(count)
    ]


def question(index: int) -> tuple[dict[str, object], str]:
    question_id = f"q-{index:03d}"
    script = proof_script(question_id)
    return (
        {
            "fact_id": f"fact-{index:03d}",
            "question_id": question_id,
            "question": f"Why does behavior {index} cross components?",
            "category": "why_design_rationale",
            "golden_answer": {"canonical": f"Reference answer {index}."},
            "golden_proof": {
                "path": (f"golden-proofs/{INSTANCE_ID}/{question_id}.sh"),
                "format": "bash",
                "sha256": hashlib.sha256(script.encode()).hexdigest(),
                "timeout_seconds": 30,
                "max_output_bytes": 65536,
            },
            "evaluation_mode": "proof_judged_v1",
            "evidence_hops": hops(),
            "reasoning_chain": reasoning(),
        },
        script,
    )


def make_bundle(count: int = 2) -> GenerationBundle:
    questions = []
    proofs = {}
    facts = []
    for index in range(1, count + 1):
        value, script = question(index)
        questions.append(value)
        proofs[value["golden_proof"]["path"]] = script
        facts.append(
            {
                "fact_id": value["fact_id"],
                "fact": f"Reference fact {index}.",
            }
        )
    return GenerationBundle.from_dict(
        {
            "schema_version": "2.0",
            "instance_id": INSTANCE_ID,
            "documentation": "Repository documentation.",
            "facts": facts,
            "questions": questions,
            "golden_proofs": proofs,
        }
    )


def replacement(record, script: str) -> dict[str, object]:
    value = record.to_dict()
    return {
        "instance_id": value["instance_id"],
        "schema_version": value["schema_version"],
        "evaluation_mode": value["evaluation_mode"],
        "question_id": value["question_id"],
        "fact_id": value["fact_id"],
        "question": value["question"],
        "category": value["category"],
        "golden_answer": value["golden_answer"],
        "evidence_hops": value["evidence_hops"],
        "reasoning_chain": value["reasoning_chain"],
        "golden_proof": {
            **value["golden_proof"],
            "sha256": hashlib.sha256(script.encode()).hexdigest(),
        },
        "golden_proof_script": script,
    }


def write_response(
    path,
    request: dict[str, object],
    replacements: list[dict[str, object]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "protocol": QA_QUESTION_REPLACEMENTS_PROTOCOL,
                "request_protocol": request["protocol"],
                "instance_id": request["instance_id"],
                "attempt": request["attempt"],
                "replacements": replacements,
            }
        )
        + "\n"
    )


def validation(patch_stable):
    return GoldenProofValidation(
        operationally_valid=True,
        repository_dependent=True,
        deterministic=True,
        repository_state_unchanged=True,
        patch_stable=patch_stable,
    )


@pytest.mark.parametrize(
    ("patch_stable", "legacy_result", "strict_result"),
    [
        (True, True, True),
        (False, True, False),
        (None, True, False),
    ],
)
def test_patch_stability_policy(
    patch_stable,
    legacy_result,
    strict_result,
):
    record = replace(
        make_bundle(1).questions[0],
        golden_proof_validation=validation(patch_stable),
    )

    assert golden_proof_validation_passes(record) is legacy_result
    assert (
        golden_proof_validation_passes(
            record,
            require_patch_stable=True,
        )
        is strict_result
    )


def test_legacy_absence_round_trips_without_patch_stable():
    record = replace(
        make_bundle(1).questions[0],
        golden_proof_validation=validation(None),
    )
    value = record.to_dict()

    assert "patch_stable" not in value["golden_proof_validation"]
    restored = type(record).from_dict(value)
    assert restored.golden_proof_validation.patch_stable is None

    execution_value = {
        "protocol": "qa_proof_execution/v1",
        "question_id": "q-001",
        "proof_kind": "golden",
        "source_sha256": "0" * 64,
        "submitted": True,
        "executed": True,
        "success": True,
        "exit_code": 0,
        "timed_out": False,
        "output_truncated": False,
        "repository_state_unchanged": True,
        "evidence_valid": True,
    }
    execution = ProofExecutionRecord.from_dict(execution_value)
    assert execution.patch_stable is None
    assert "patch_stable" not in execution.to_dict()

    execution_value["patch_stable"] = True
    assert ProofExecutionRecord.from_dict(execution_value).patch_stable is True


def test_replaces_question_and_proof_atomically(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    request_path = tmp_path / "request.json"
    request = build_question_replacement_request(
        bundle_dir,
        request_path,
        mode="regeneration",
        question_ids=["q-001"],
        attempt=1,
        max_attempts=2,
    )
    record = load_qa_records(bundle_dir)[0]
    script = proof_script("replacement")
    value = replacement(record, script)
    value["question"] = "How does the replacement cross components?"
    value["golden_answer"] = {"canonical": "Replacement answer."}
    value["evidence_hops"] = hops(4)
    value["reasoning_chain"] = reasoning(4)
    response_path = tmp_path / "response.json"
    write_response(response_path, request, [value])
    proof_path = bundle_dir / record.golden_proof.path
    questions_path = bundle_dir / "questions.jsonl"
    original_write_text = Path.write_text

    def guarded_write_text(path, data, *args, **kwargs):
        if path in {proof_path, questions_path}:
            raise AssertionError(f"atomic replacement rewrote {path}")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded_write_text)

    summary = apply_question_replacements(
        bundle_dir,
        request_path,
        response_path,
    )

    assert summary["n_applied"] == 1
    loaded = load_generation_bundle(bundle_dir)
    replaced = loaded.questions[0]
    assert replaced.question == value["question"]
    assert replaced.golden_answer.canonical == "Replacement answer."
    assert len(replaced.evidence_hops) == 4
    assert proof_path.read_text() == script
    assert proof_path.stat().st_mode & 0o111
    assert replaced.golden_proof.sha256 == hashlib.sha256(proof_path.read_bytes()).hexdigest()
    assert load_generation_bundle(bundle_dir).questions[0] == replaced


def test_eperm_fallback_writes_and_verifies_targets(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    request_path = tmp_path / "request.json"
    request = build_question_replacement_request(
        bundle_dir,
        request_path,
        mode="regeneration",
        question_ids=["q-001"],
        attempt=1,
        max_attempts=2,
    )
    record = load_qa_records(bundle_dir)[0]
    script = proof_script("replacement")
    value = replacement(record, script)
    value["question"] = "How does the fallback cross components?"
    response_path = tmp_path / "response.json"
    write_response(response_path, request, [value])
    original_path_replace = Path.replace

    def reject_questions_replace(source, target):
        if source.name.startswith(".questions.jsonl."):
            raise PermissionError(errno.EPERM, "mounted filesystem")
        return original_path_replace(source, target)

    def reject_proof_replace(source, target):
        raise PermissionError(errno.EPERM, "mounted filesystem")

    monkeypatch.setattr(Path, "replace", reject_questions_replace)
    monkeypatch.setattr(qa_replacement.os, "replace", reject_proof_replace)

    summary = apply_question_replacements(
        bundle_dir,
        request_path,
        response_path,
    )

    assert summary["n_applied"] == 1
    assert load_qa_records(bundle_dir)[0].question == value["question"]
    assert (bundle_dir / record.golden_proof.path).read_text() == script
    assert not list(bundle_dir.glob(".*.tmp"))


def test_rejects_malformed_unknown_and_duplicate_ids(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    request_path = tmp_path / "request.json"
    request = build_question_replacement_request(
        bundle_dir,
        request_path,
        mode="repair",
        question_ids=["q-001"],
        attempt=1,
        max_attempts=2,
    )
    record = load_qa_records(bundle_dir)[0]
    valid = replacement(record, proof_script("replacement"))
    response_path = tmp_path / "response.json"

    malformed = dict(valid)
    malformed["question_id"] = 1
    write_response(response_path, request, [malformed])
    with pytest.raises(ValueError, match="malformed ID"):
        apply_question_replacements(
            bundle_dir,
            request_path,
            response_path,
        )

    unknown = dict(valid)
    unknown["question_id"] = "q-999"
    write_response(response_path, request, [unknown])
    with pytest.raises(ValueError, match="Unknown replacement"):
        apply_question_replacements(
            bundle_dir,
            request_path,
            response_path,
        )

    write_response(response_path, request, [valid, valid])
    with pytest.raises(ValueError, match="Duplicate replacement"):
        apply_question_replacements(
            bundle_dir,
            request_path,
            response_path,
        )


def test_collision_pruning_uses_existing_hardness_policy(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    request_path = tmp_path / "request.json"
    request = build_question_replacement_request(
        bundle_dir,
        request_path,
        mode="regeneration",
        question_ids=["q-001"],
        attempt=1,
        max_attempts=1,
    )
    record = load_qa_records(bundle_dir)[0]
    value = replacement(record, proof_script("replacement"))
    value["fact_id"] = "fact-002"
    value["evidence_hops"] = hops(4)
    value["reasoning_chain"] = reasoning(4)
    response_path = tmp_path / "response.json"
    write_response(response_path, request, [value])

    summary = apply_question_replacements(
        bundle_dir,
        request_path,
        response_path,
    )

    assert summary["collision_drops"][0]["question_id"] == "q-002"
    assert [record.question_id for record in load_qa_records(bundle_dir)] == ["q-001"]


def test_invalid_attempt_preserves_siblings_and_target_for_retry(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(3), bundle_dir)
    request_path = tmp_path / "request.json"
    request = build_question_replacement_request(
        bundle_dir,
        request_path,
        mode="repair",
        question_ids=["q-001"],
        attempt=1,
        max_attempts=2,
    )
    record = load_qa_records(bundle_dir)[0]
    value = replacement(record, proof_script("replacement"))
    value["golden_proof"]["sha256"] = "0" * 64
    response_path = tmp_path / "response.json"
    write_response(response_path, request, [value])

    summary = apply_question_replacements(
        bundle_dir,
        request_path,
        response_path,
    )

    assert summary["n_retained_for_retry"] == 1
    assert [record.question_id for record in load_qa_records(bundle_dir)] == [
        "q-001",
        "q-002",
        "q-003",
    ]


def test_exhaustion_drops_only_failed_question(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(3), bundle_dir)
    request_path = tmp_path / "request.json"
    request = build_question_replacement_request(
        bundle_dir,
        request_path,
        mode="repair",
        question_ids=["q-001"],
        attempt=2,
        max_attempts=2,
    )
    response_path = tmp_path / "response.json"
    write_response(response_path, request, [])

    summary = apply_question_replacements(
        bundle_dir,
        request_path,
        response_path,
    )

    assert summary["n_dropped"] == 1
    assert [record.question_id for record in load_qa_records(bundle_dir)] == [
        "q-002",
        "q-003",
    ]
    history = json.loads((bundle_dir / "question-replacement-attempts.json").read_text())
    assert history["attempts"][0]["diagnostics"] == {}
    assert history["attempts"][0]["results"][0]["status"] == ("dropped_exhausted")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_id", "other__issue"),
        ("schema_version", "1.0"),
        ("evaluation_mode", "other"),
        ("question_id", "q-999"),
    ],
)
def test_immutable_identity_fields_cannot_change(
    tmp_path,
    field,
    value,
):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    request_path = tmp_path / "request.json"
    request = build_question_replacement_request(
        bundle_dir,
        request_path,
        mode="repair",
        question_ids=["q-001"],
        attempt=1,
        max_attempts=2,
    )
    record = load_qa_records(bundle_dir)[0]
    replacement_value = replacement(record, proof_script("replacement"))
    replacement_value[field] = value
    response_path = tmp_path / "response.json"
    write_response(response_path, request, [replacement_value])

    if field == "question_id":
        with pytest.raises(ValueError, match="Unknown replacement"):
            apply_question_replacements(
                bundle_dir,
                request_path,
                response_path,
            )
        return
    summary = apply_question_replacements(
        bundle_dir,
        request_path,
        response_path,
    )
    assert summary["results"][0]["status"] == "retained_for_retry"
    assert f"immutable field {field}" in summary["results"][0]["error"]


def test_detailed_post_patch_feedback_reaches_replacement_request(
    tmp_path,
):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    state = {
        "passed": False,
        "exit_code": 1,
        "timed_out": False,
        "output_truncated": False,
        "failure_codes": ["nonzero_exit"],
    }
    diagnostics = {
        "protocol": QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL,
        "instance_id": INSTANCE_ID,
        "questions": [
            {
                "question_id": "q-001",
                "pristine": {
                    **state,
                    "repository_state_unchanged": True,
                    "stdout": '{"protocol":"qa_proof/v1"}\n',
                    "stderr": "",
                },
                "post_patch": {
                    **state,
                    "applied": True,
                    "repository_state_unchanged": False,
                    "stdout": "",
                    "stderr": "assertion failed after patch\n",
                    "patch_application": {
                        "exit_code": 0,
                        "output_truncated": False,
                        "stdout": "Applied patch\n",
                        "stderr": "",
                        "failure_codes": [],
                    },
                },
            }
        ],
    }
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))
    request_path = tmp_path / "request.json"

    request = build_question_replacement_request(
        bundle_dir,
        request_path,
        mode="regeneration",
        question_ids=["q-001"],
        attempt=1,
        max_attempts=2,
        diagnostics_path=diagnostics_path,
    )

    assert set(request["diagnostics"]["q-001"]) == {
        "pristine",
        "post_patch",
    }
    assert request["diagnostics"]["q-001"]["post_patch"] == {
        **state,
        "applied": True,
        "repository_state_unchanged": False,
        "stdout": "",
        "stderr": "assertion failed after patch\n",
        "patch_application": {
            "exit_code": 0,
            "output_truncated": False,
            "stdout": "Applied patch\n",
            "stderr": "",
            "failure_codes": [],
        },
    }


def test_unknown_post_patch_repository_state_reaches_replacement_request(
    tmp_path,
):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    state = {
        "passed": False,
        "exit_code": None,
        "timed_out": False,
        "output_truncated": False,
        "failure_codes": ["patch_apply_failed"],
    }
    diagnostics = {
        "protocol": QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL,
        "instance_id": INSTANCE_ID,
        "questions": [
            {
                "question_id": "q-001",
                "pristine": state,
                "post_patch": {
                    **state,
                    "applied": False,
                    "repository_state_unchanged": None,
                },
            }
        ],
    }
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))

    request = build_question_replacement_request(
        bundle_dir,
        tmp_path / "request.json",
        mode="regeneration",
        question_ids=["q-001"],
        attempt=1,
        max_attempts=2,
        diagnostics_path=diagnostics_path,
    )

    assert (
        request["diagnostics"]["q-001"]["post_patch"][
            "repository_state_unchanged"
        ]
        is None
    )


@pytest.mark.parametrize(
    "private_field",
    ["gold_patch", "patch", "patch_source", "source"],
)
def test_private_patch_fields_are_rejected(tmp_path, private_field):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    state = {
        "passed": False,
        "exit_code": 1,
        "timed_out": False,
        "output_truncated": False,
        "failure_codes": ["post_patch_validation_failed"],
    }
    diagnostics = {
        "protocol": QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL,
        "instance_id": INSTANCE_ID,
        "questions": [
            {
                "question_id": "q-001",
                "pristine": state,
                "post_patch": {
                    **state,
                    private_field: "private patch material",
                },
            }
        ],
    }
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))
    request_path = tmp_path / "request.json"

    with pytest.raises(ValueError, match="unknown keys"):
        build_question_replacement_request(
            bundle_dir,
            request_path,
            mode="regeneration",
            question_ids=["q-001"],
            attempt=1,
            max_attempts=2,
            diagnostics_path=diagnostics_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exit_code", True, "exit_code"),
        ("output_truncated", None, "output_truncated"),
        ("stdout", 1, "stdout"),
        ("stderr", [], "stderr"),
        ("failure_codes", ["failed", "failed"], "duplicates"),
    ],
)
def test_malformed_patch_application_diagnostics_are_rejected(
    tmp_path,
    field,
    value,
    message,
):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    patch_application = {
        "exit_code": 1,
        "output_truncated": False,
        "stdout": "",
        "stderr": "patch failed",
        "failure_codes": ["patch_application_failed"],
    }
    patch_application[field] = value
    diagnostics = {
        "protocol": QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL,
        "instance_id": INSTANCE_ID,
        "questions": [
            {
                "question_id": "q-001",
                "pristine": {
                    "passed": True,
                    "exit_code": 0,
                    "timed_out": False,
                    "output_truncated": False,
                    "failure_codes": [],
                },
                "post_patch": {
                    "passed": False,
                    "applied": False,
                    "exit_code": None,
                    "timed_out": False,
                    "output_truncated": False,
                    "failure_codes": ["patch_application_failed"],
                    "patch_application": patch_application,
                },
            }
        ],
    }
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))

    with pytest.raises(ValueError, match=message):
        build_question_replacement_request(
            bundle_dir,
            tmp_path / "request.json",
            mode="regeneration",
            question_ids=["q-001"],
            attempt=1,
            max_attempts=2,
            diagnostics_path=diagnostics_path,
        )


@pytest.mark.parametrize("private_location", ["field", "stdout", "stderr"])
def test_private_patch_application_content_is_rejected(
    tmp_path,
    private_location,
):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    patch_application = {
        "exit_code": 1,
        "output_truncated": False,
        "stdout": "",
        "stderr": "",
        "failure_codes": ["patch_application_failed"],
    }
    if private_location == "field":
        patch_application["patch"] = "private"
    else:
        patch_application[private_location] = "diff --git a/private b/private"
    diagnostics = {
        "protocol": QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL,
        "instance_id": INSTANCE_ID,
        "questions": [
            {
                "question_id": "q-001",
                "pristine": {
                    "passed": True,
                    "exit_code": 0,
                    "timed_out": False,
                    "output_truncated": False,
                    "failure_codes": [],
                },
                "post_patch": {
                    "passed": False,
                    "applied": False,
                    "exit_code": None,
                    "timed_out": False,
                    "output_truncated": False,
                    "failure_codes": ["patch_application_failed"],
                    "patch_application": patch_application,
                },
            }
        ],
    }
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))

    with pytest.raises(
        ValueError,
        match=r"unknown keys|private patch content",
    ):
        build_question_replacement_request(
            bundle_dir,
            tmp_path / "request.json",
            mode="regeneration",
            question_ids=["q-001"],
            attempt=1,
            max_attempts=2,
            diagnostics_path=diagnostics_path,
        )


@pytest.mark.parametrize(
    "output",
    [
        "diff --git a/private.py b/private.py\n",
        "failure\n--- a/private.py\n+++ b/private.py\n@@ -1 +1 @@\n",
    ],
)
def test_private_patch_content_in_diagnostic_output_is_rejected(
    tmp_path,
    output,
):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    state = {
        "passed": False,
        "exit_code": 1,
        "timed_out": False,
        "output_truncated": False,
        "failure_codes": ["post_patch_validation_failed"],
    }
    diagnostics = {
        "protocol": QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL,
        "instance_id": INSTANCE_ID,
        "questions": [
            {
                "question_id": "q-001",
                "pristine": state,
                "post_patch": {
                    **state,
                    "stderr": output,
                },
            }
        ],
    }
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))

    with pytest.raises(ValueError, match="private patch content"):
        build_question_replacement_request(
            bundle_dir,
            tmp_path / "request.json",
            mode="regeneration",
            question_ids=["q-001"],
            attempt=1,
            max_attempts=2,
            diagnostics_path=diagnostics_path,
        )


def test_diagnostic_output_is_bounded(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle(), bundle_dir)
    state = {
        "passed": False,
        "exit_code": 1,
        "timed_out": False,
        "output_truncated": True,
        "failure_codes": ["output_truncated"],
    }
    diagnostics = {
        "protocol": QA_DUAL_STATE_DIAGNOSTICS_PROTOCOL,
        "instance_id": INSTANCE_ID,
        "questions": [
            {
                "question_id": "q-001",
                "pristine": state,
                "post_patch": {
                    **state,
                    "stdout": "x" * 65537,
                },
            }
        ],
    }
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))

    with pytest.raises(ValueError, match="65536 UTF-8 bytes"):
        build_question_replacement_request(
            bundle_dir,
            tmp_path / "request.json",
            mode="regeneration",
            question_ids=["q-001"],
            attempt=1,
            max_attempts=2,
            diagnostics_path=diagnostics_path,
        )
