from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from interactive_benchmark_generation.dataset_io import write_json
from interactive_benchmark_generation.qa_io import (
    load_facts,
    load_generation_bundle,
    load_qa_records,
    replace_questions,
    validate_record_relationships,
    write_qa_bundle,
)
from interactive_benchmark_generation.qa_models import (
    QA_SCHEMA_V2,
    AdversarialReview,
    AnswerJudgeAssessment,
    GenerationBundle,
    GoldenProofValidation,
    ProbeValidation,
    ProofExecutionRecord,
    ProofJudgeAssessment,
    QARecord,
)

ADVERSARIAL_REVIEW_CHECKS_V1 = {
    "repository_grounded",
    "category_fit",
    "multi_hop_required",
    "single_lookup_resistant",
    "evidence_valid",
    "causal_chain_valid",
    "answer_unambiguous",
    "answer_contract_valid",
    "single_coherent_question",
    "alternative_grounded_answer_absent",
}
ADVERSARIAL_REVIEW_CHECKS_V2 = {
    "repository_grounded",
    "category_fit",
    "multi_hop_required",
    "single_lookup_resistant",
    "evidence_valid",
    "reasoning_chain_valid",
    "golden_proof_operationally_valid",
    "golden_proof_relevant",
    "golden_proof_non_circular",
    "answer_unambiguous",
    "single_coherent_question",
    "alternative_answers_accounted_for",
}
ADVERSARIAL_REVIEW_CHECKS = ADVERSARIAL_REVIEW_CHECKS_V1


def golden_proof_validation_passes(
    record: QARecord,
    *,
    require_patch_stable: bool = False,
) -> bool:
    validation = record.golden_proof_validation
    return (
        validation is not None
        and validation.operationally_valid
        and validation.repository_dependent
        and validation.deterministic
        and validation.repository_state_unchanged
        and (not require_patch_stable or validation.patch_stable is True)
    )


def _report_items(report: object, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        raise ValueError("Imported report must contain a JSON object")
    for field in fields:
        values = report.get(field)
        if isinstance(values, list) and all(isinstance(value, dict) for value in values):
            return values
    raise ValueError(f"Imported report must contain one of these lists: {', '.join(fields)}")


def _proof_execution_values(report: object) -> list[dict[str, Any]]:
    values = _report_items(report, ("executions", "results", "records"))
    flattened = []
    for value in values:
        if "proof_kind" in value:
            flattened.append(value)
            continue
        question_id = value.get("question_id")
        for proof_kind in ("golden", "agent"):
            nested = value.get(proof_kind)
            if isinstance(nested, dict):
                flattened.append(
                    {
                        "question_id": question_id,
                        "proof_kind": proof_kind,
                        **nested,
                    }
                )
    return flattened


def import_proof_execution_report(
    bundle_dir: Path,
    report_path: Path,
    stage: str = "combined",
) -> dict[str, Any]:
    if stage not in {"golden", "agent", "combined"}:
        raise ValueError("Proof execution stage must be golden, agent, or combined")
    report = json.loads(report_path.read_text())
    records = load_qa_records(bundle_dir)
    if not records or any(record.schema_version != QA_SCHEMA_V2 for record in records):
        raise ValueError("Proof execution imports require a v2 QA bundle")
    expected_ids = {record.question_id for record in records}
    parsed = [
        ProofExecutionRecord.from_dict(value)
        for value in _proof_execution_values(report)
    ]
    by_key = {}
    for execution in parsed:
        key = (execution.question_id, execution.proof_kind.value)
        if key in by_key:
            raise ValueError(f"Duplicate proof execution: {key}")
        if execution.question_id not in expected_ids:
            raise ValueError(f"Unknown proof execution question_id: {execution.question_id}")
        by_key[key] = execution
    expected_kinds = {
        "golden": ("golden",),
        "agent": ("agent",),
        "combined": ("golden", "agent"),
    }[stage]
    expected_keys = {
        (question_id, proof_kind)
        for question_id in expected_ids
        for proof_kind in expected_kinds
    }
    if set(by_key) != expected_keys:
        raise ValueError(
            "Proof execution keys differ: "
            f"missing={sorted(expected_keys - set(by_key))}, "
            f"unknown={sorted(set(by_key) - expected_keys)}"
        )
    records_by_id = {record.question_id: record for record in records}
    if stage in {"golden", "combined"}:
        for question_id in expected_ids:
            golden = by_key[(question_id, "golden")]
            expected_hash = records_by_id[question_id].golden_proof.sha256
            if golden.source_sha256 != expected_hash:
                raise ValueError(f"Golden proof execution hash differs for {question_id}")
    updated_records = records
    if stage == "golden":
        updated_records = []
        for record in records:
            execution = by_key[(record.question_id, "golden")]
            if execution.repository_dependent is None:
                raise ValueError(
                    f"Golden proof execution lacks repository_dependent for "
                    f"{record.question_id}"
                )
            if execution.deterministic is None:
                raise ValueError(
                    f"Golden proof execution lacks deterministic for "
                    f"{record.question_id}"
                )
            validation = GoldenProofValidation(
                operationally_valid=(
                    execution.success
                    and execution.repository_dependent
                    and execution.deterministic
                    and execution.repository_state_unchanged
                ),
                repository_dependent=execution.repository_dependent,
                deterministic=execution.deterministic,
                repository_state_unchanged=execution.repository_state_unchanged,
                patch_stable=execution.patch_stable,
            )
            updated_records.append(
                replace(record, golden_proof_validation=validation)
            )
        replace_questions(bundle_dir, updated_records)
    imported = {
        "protocol": "qa_proof_execution_report/v1",
        "source": str(report_path),
        "stage": stage,
        "n_questions": len(expected_ids),
        "n_executions": len(parsed),
        "executions": [execution.to_dict() for execution in parsed],
    }
    if stage == "golden":
        write_json(bundle_dir / "golden-proof-executions.json", imported)
    elif stage == "agent":
        golden_path = bundle_dir / "golden-proof-executions.json"
        if not golden_path.is_file():
            raise ValueError(
                "Agent proof import requires an earlier golden proof execution import"
            )
        combined = [*load_proof_executions(golden_path).values(), *parsed]
        combined_report = {
            **imported,
            "stage": "combined",
            "n_executions": len(combined),
            "executions": [execution.to_dict() for execution in combined],
        }
        write_json(bundle_dir / "proof-executions.json", combined_report)
        imported = combined_report
    else:
        write_json(bundle_dir / "proof-executions.json", imported)
    return imported


def import_golden_proof_execution_report(
    bundle_dir: Path,
    report_path: Path,
    *,
    drop_invalid: bool = False,
    require_patch_stable: bool = False,
) -> dict[str, Any]:
    """Attach trusted golden execution results to each question.

    `drop_invalid` also removes the questions whose golden proof failed trusted
    execution. `filter_validated_bundle` would remove them anyway, but only after
    review — and a reviewer that contradicts the trusted status on one of them
    makes `import_adversarial_review` refuse the whole report, taking the issue's
    good questions with it. Removing them first means the reviewer never sees a
    question it could contradict.
    """
    imported = import_proof_execution_report(
        bundle_dir,
        report_path,
        stage="golden",
    )
    if not drop_invalid:
        return imported
    records = load_qa_records(bundle_dir)
    kept = [
        record
        for record in records
        if golden_proof_validation_passes(
            record,
            require_patch_stable=require_patch_stable,
        )
    ]
    if not kept:
        raise ValueError(
            "No golden proof passed trusted execution: "
            + ", ".join(record.question_id for record in records)
        )
    if len(kept) != len(records):
        replace_questions(bundle_dir, kept)
    return {
        **imported,
        "n_questions_kept": len(kept),
        "n_questions_dropped": len(records) - len(kept),
        "dropped_question_ids": [
            record.question_id for record in records if record not in kept
        ],
    }


def load_proof_executions(path: Path) -> dict[tuple[str, str], ProofExecutionRecord]:
    report = json.loads(path.read_text())
    executions = [
        ProofExecutionRecord.from_dict(value)
        for value in _proof_execution_values(report)
    ]
    result = {}
    for execution in executions:
        key = (execution.question_id, execution.proof_kind.value)
        if key in result:
            raise ValueError(f"Duplicate proof execution: {key}")
        result[key] = execution
    return result


def import_judge_report(
    bundle_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    values = _report_items(report, ("assessments", "results", "judgements", "judgments"))
    assessments = [ProofJudgeAssessment.from_dict(value) for value in values]
    records = load_qa_records(bundle_dir)
    expected_ids = {record.question_id for record in records}
    by_id = {}
    for assessment in assessments:
        if assessment.question_id in by_id:
            raise ValueError(f"Duplicate judge assessment: {assessment.question_id}")
        by_id[assessment.question_id] = assessment
    if set(by_id) != expected_ids:
        raise ValueError(
            f"Judge IDs differ: missing={sorted(expected_ids - set(by_id))}, "
            f"unknown={sorted(set(by_id) - expected_ids)}"
        )
    imported = {
        "protocol": "qa_proof_judge_report/v1",
        "source": str(report_path),
        "model": report.get("model"),
        "reasoning_effort": report.get("reasoning_effort"),
        "cost": report.get("cost", 0),
        "assessments": [assessment.to_dict() for assessment in assessments],
    }
    write_json(bundle_dir / "judge-assessments.json", imported)
    return imported


def import_answer_judge_report(
    bundle_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    if report.get("protocol") != "qa_answer_judge_report/v1":
        raise ValueError("Answer judge report protocol is invalid")
    values = _report_items(
        report,
        ("assessments", "results", "judgements", "judgments"),
    )
    assessments = [
        AnswerJudgeAssessment.from_dict(value) for value in values
    ]
    records = load_qa_records(bundle_dir)
    expected_ids = {record.question_id for record in records}
    by_id = {}
    for assessment in assessments:
        if assessment.question_id in by_id:
            raise ValueError(
                f"Duplicate answer judge assessment: "
                f"{assessment.question_id}"
            )
        by_id[assessment.question_id] = assessment
    if set(by_id) != expected_ids:
        raise ValueError(
            f"Answer judge IDs differ: "
            f"missing={sorted(expected_ids - set(by_id))}, "
            f"unknown={sorted(set(by_id) - expected_ids)}"
        )
    imported = {
        "protocol": "qa_answer_judge_report/v1",
        "source": str(report_path),
        "model": report.get("model"),
        "reasoning_effort": report.get("reasoning_effort"),
        "cost": report.get("cost", 0),
        "assessments": [
            assessment.to_dict() for assessment in assessments
        ],
    }
    write_json(bundle_dir / "answer-judge-assessments.json", imported)
    return imported


def load_judge_assessments(
    path: Path,
) -> tuple[
    dict[str, ProofJudgeAssessment | AnswerJudgeAssessment],
    dict[str, Any],
]:
    report = json.loads(path.read_text())
    values = _report_items(report, ("assessments", "results", "judgements", "judgments"))
    protocol = report.get("protocol", "qa_proof_judge_report/v1")
    if protocol == "qa_answer_judge_report/v1":
        assessments = [
            AnswerJudgeAssessment.from_dict(value) for value in values
        ]
    elif protocol == "qa_proof_judge_report/v1":
        assessments = [
            ProofJudgeAssessment.from_dict(value) for value in values
        ]
    else:
        raise ValueError("Judge report protocol is invalid")
    by_id = {}
    for assessment in assessments:
        if assessment.question_id in by_id:
            raise ValueError(f"Duplicate judge assessment: {assessment.question_id}")
        by_id[assessment.question_id] = assessment
    return by_id, report


def repository_state(repository: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Could not inspect repository state: {result.stderr.strip()}")
    return result.stdout


def _run_probe(
    probe_path: Path,
    answer: str,
    expected_output: str,
    repository: Path,
    timeout: float,
    initial_state: str,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            [str(probe_path), answer],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = time.monotonic() - started
        output_exact = result.stdout in {expected_output, f"{expected_output}\n"}
        state_unchanged = repository_state(repository) == initial_state
        passed = result.returncode == 0 and output_exact and state_unchanged
        return {
            "answer": answer,
            "expected_output": expected_output,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timed_out": False,
            "state_unchanged": state_unchanged,
            "duration_seconds": duration,
            "passed": passed,
        }
    except subprocess.TimeoutExpired as error:
        duration = time.monotonic() - started
        return {
            "answer": answer,
            "expected_output": expected_output,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "returncode": None,
            "timed_out": True,
            "state_unchanged": repository_state(repository) == initial_state,
            "duration_seconds": duration,
            "passed": False,
        }
    except OSError as error:
        return {
            "answer": answer,
            "expected_output": expected_output,
            "stdout": "",
            "stderr": str(error),
            "returncode": None,
            "timed_out": False,
            "state_unchanged": repository_state(repository) == initial_state,
            "duration_seconds": time.monotonic() - started,
            "passed": False,
        }


def validate_probe(
    bundle_dir: Path,
    record: QARecord,
    repository: Path,
    timeout: float,
) -> tuple[QARecord, dict[str, Any]]:
    probe_path = (bundle_dir / record.probe_path).resolve()
    bundle_root = bundle_dir.resolve()
    if bundle_root not in probe_path.parents or not probe_path.is_file():
        raise ValueError(f"Missing probe for {record.question_id}: {record.probe_path}")
    initial_state = repository_state(repository)
    runs = [
        _run_probe(probe_path, answer, "PASSED", repository, timeout, initial_state)
        for answer in record.accepted_answers
    ]
    runs.extend(
        _run_probe(probe_path, answer, "FAILED", repository, timeout, initial_state)
        for answer in record.incorrect_answers
    )
    accepted_count = len(record.accepted_answers)
    accepted_passed = all(run["passed"] for run in runs[:accepted_count])
    incorrect_rejected = all(run["passed"] for run in runs[accepted_count:])
    validation = ProbeValidation(
        accepted_passed=accepted_passed,
        incorrect_rejected=incorrect_rejected,
        runs=tuple(runs),
    )
    updated_record = replace(record, probe_validation=validation)
    return updated_record, {
        "question_id": record.question_id,
        "valid": accepted_passed and incorrect_rejected,
        "accepted_passed": accepted_passed,
        "incorrect_rejected": incorrect_rejected,
        "runs": runs,
    }


def validate_qa_bundle(
    bundle_dir: Path,
    repository: Path,
    timeout: float = 30.0,
) -> dict[str, Any]:
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    bundle_dir = bundle_dir.resolve()
    repository = repository.resolve()
    facts = load_facts(bundle_dir)
    records = load_qa_records(bundle_dir)
    validate_record_relationships(facts, records)
    initial_state = repository_state(repository)
    updated_records = []
    results = []
    for record in records:
        updated_record, result = validate_probe(bundle_dir, record, repository, timeout)
        updated_records.append(updated_record)
        results.append(result)
    final_state_unchanged = repository_state(repository) == initial_state
    valid_count = sum(result["valid"] for result in results)
    report = {
        "valid": valid_count == len(results) and final_state_unchanged,
        "n_questions": len(results),
        "n_valid": valid_count,
        "valid_proportion": valid_count / len(results) if results else 0.0,
        "repository_state_unchanged": final_state_unchanged,
        "questions": results,
    }
    replace_questions(bundle_dir, updated_records)
    write_json(bundle_dir / "probe-validation.json", report)
    debug_path = bundle_dir / "debug.json"
    debug = json.loads(debug_path.read_text()) if debug_path.exists() else {}
    if not isinstance(debug, dict):
        debug = {"generation": debug}
    debug["probe_validation"] = report
    write_json(debug_path, debug)
    return report


def import_probe_validation_report(
    bundle_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError("Harbor probe report must contain a results list")
    records = load_qa_records(bundle_dir)
    results_by_id = {}
    for result in report["results"]:
        if not isinstance(result, dict) or not isinstance(result.get("question_id"), str):
            raise ValueError("Every Harbor probe result must have a question_id")
        question_id = result["question_id"]
        if question_id in results_by_id:
            raise ValueError(f"Duplicate Harbor probe result: {question_id}")
        results_by_id[question_id] = result
    expected_ids = {record.question_id for record in records}
    actual_ids = set(results_by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Harbor probe result IDs differ: missing={sorted(expected_ids - actual_ids)}, "
            f"unknown={sorted(actual_ids - expected_ids)}"
        )
    updated_records = []
    for record in records:
        result = results_by_id[record.question_id]
        accepted = result.get("accepted")
        incorrect = result.get("incorrect")
        if not isinstance(accepted, list) or not isinstance(incorrect, list):
            raise ValueError(f"Invalid Harbor probe result for {record.question_id}")
        accepted_passed = (
            [run.get("answer") for run in accepted if isinstance(run, dict)]
            == list(record.accepted_answers)
            and all(isinstance(run, dict) and run.get("passed") is True for run in accepted)
            and result.get("repository_dependent", True) is True
        )
        incorrect_rejected = (
            [run.get("answer") for run in incorrect if isinstance(run, dict)]
            == list(record.incorrect_answers)
            and all(isinstance(run, dict) and run.get("rejected") is True for run in incorrect)
        )
        runs = tuple(
            {"expected_output": "PASSED", **run} for run in accepted if isinstance(run, dict)
        ) + tuple(
            {"expected_output": "FAILED", **run} for run in incorrect if isinstance(run, dict)
        )
        validation = ProbeValidation(
            accepted_passed=accepted_passed,
            incorrect_rejected=incorrect_rejected,
            runs=runs,
        )
        updated_records.append(replace(record, probe_validation=validation))
    imported_report = {
        "valid": report.get("valid") is True
        and all(
            record.probe_validation is not None
            and record.probe_validation.accepted_passed
            and record.probe_validation.incorrect_rejected
            for record in updated_records
        ),
        "source": str(report_path),
        "n_questions": len(updated_records),
        "results": report["results"],
    }
    replace_questions(bundle_dir, updated_records)
    write_json(bundle_dir / "probe-validation.json", imported_report)
    debug_path = bundle_dir / "debug.json"
    debug = json.loads(debug_path.read_text()) if debug_path.exists() else {}
    if not isinstance(debug, dict):
        debug = {"generation": debug}
    debug["probe_validation"] = imported_report
    write_json(debug_path, debug)
    return imported_report


def import_adversarial_review(
    bundle_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    reviews = report.get("reviews") if isinstance(report, dict) else None
    if not isinstance(reviews, list):
        raise ValueError("Adversarial review report must contain a reviews list")
    records = load_qa_records(bundle_dir)
    schema_versions = {record.schema_version for record in records}
    if len(schema_versions) != 1:
        raise ValueError("Adversarial review import does not support mixed schema versions")
    schema_version = next(iter(schema_versions))
    expected_checks = (
        ADVERSARIAL_REVIEW_CHECKS_V2
        if schema_version == QA_SCHEMA_V2
        else ADVERSARIAL_REVIEW_CHECKS_V1
    )
    records_by_id = {record.question_id: record for record in records}
    if schema_version == QA_SCHEMA_V2:
        missing_validation = [
            record.question_id
            for record in records
            if record.golden_proof_validation is None
        ]
        if missing_validation:
            raise ValueError(
                "v2 adversarial review requires attached golden proof validation: "
                f"{missing_validation}"
            )
    instance_ids = {record.instance_id for record in records}
    if len(instance_ids) != 1 or report.get("instance_id") not in instance_ids:
        raise ValueError("Adversarial review instance_id differs from the QA bundle")
    reviews_by_id = {}
    for review in reviews:
        if not isinstance(review, dict) or not isinstance(review.get("question_id"), str):
            raise ValueError("Every adversarial review must have a question_id")
        question_id = review["question_id"]
        if question_id in reviews_by_id:
            raise ValueError(f"Duplicate adversarial review: {question_id}")
        checks = review.get("checks")
        if not isinstance(checks, dict) or set(checks) != expected_checks:
            raise ValueError(f"Invalid adversarial checks for {question_id}")
        if schema_version == QA_SCHEMA_V2:
            record = records_by_id.get(question_id)
            trusted_status = (
                record.golden_proof_validation.operationally_valid
                if record is not None and record.golden_proof_validation is not None
                else None
            )
            reviewer_status = checks["golden_proof_operationally_valid"]
            if reviewer_status != trusted_status:
                raise ValueError(
                    "golden_proof_operationally_valid disagrees with trusted "
                    f"validation for {question_id}: "
                    f"reviewer={reviewer_status}, trusted={trusted_status}"
                )
        accepted = review.get("accepted")
        rejection_codes = review.get("rejection_codes")
        defeating_paths = review.get("defeating_paths")
        if accepted is False and (
            not isinstance(rejection_codes, list)
            or not rejection_codes
            or not isinstance(defeating_paths, list)
            or not defeating_paths
        ):
            raise ValueError(
                f"Rejected review lacks structured diagnostics for {question_id}"
            )
        if accepted is True and (rejection_codes or defeating_paths):
            raise ValueError(
                f"Accepted review contains rejection diagnostics for {question_id}"
            )
        reviews_by_id[question_id] = review
    expected_ids = {record.question_id for record in records}
    actual_ids = set(reviews_by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Adversarial review IDs differ: missing={sorted(expected_ids - actual_ids)}, "
            f"unknown={sorted(actual_ids - expected_ids)}"
        )
    updated_records = []
    for record in records:
        value = reviews_by_id[record.question_id]
        review = AdversarialReview.from_dict(value)
        updated_records.append(replace(record, adversarial_review=review))
    imported = {
        "valid": all(
            record.adversarial_review is not None and record.adversarial_review.accepted
            for record in updated_records
        ),
        "source": str(report_path),
        "schema_version": schema_version,
        "n_questions": len(updated_records),
        "n_accepted": sum(
            record.adversarial_review is not None and record.adversarial_review.accepted
            for record in updated_records
        ),
        "reviews": reviews,
    }
    replace_questions(bundle_dir, updated_records)
    write_json(bundle_dir / "adversarial-review.json", imported)
    return imported


def filter_validated_bundle(
    bundle_dir: Path,
    output_dir: Path,
    *,
    require_patch_stable: bool = False,
) -> dict[str, Any]:
    bundle = load_generation_bundle(bundle_dir)
    if bundle.schema_version == QA_SCHEMA_V2:
        records = tuple(
            record
            for record in bundle.questions
            if golden_proof_validation_passes(
                record,
                require_patch_stable=require_patch_stable,
            )
            and record.adversarial_review is not None
            and record.adversarial_review.accepted
            and set(record.adversarial_review.checks)
            == ADVERSARIAL_REVIEW_CHECKS_V2
        )
    else:
        review_required = any(
            record.adversarial_review is not None for record in bundle.questions
        )
        records = tuple(
            record
            for record in bundle.questions
            if record.probe_validation is not None
            and record.probe_validation.accepted_passed
            and record.probe_validation.incorrect_rejected
            and (
                not review_required
                or (
                    record.adversarial_review is not None
                    and record.adversarial_review.accepted
                )
            )
        )
    summary = {
        "instance_id": bundle.instance_id,
        "n_candidates": len(bundle.questions),
        "n_valid": len(records),
        "n_rejected": len(bundle.questions) - len(records),
    }
    if not records:
        return summary
    fact_ids = {record.fact_id for record in records}
    probe_paths = {record.probe_path for record in records}
    golden_proof_paths = {
        record.golden_proof.path
        for record in records
        if record.golden_proof is not None
    }
    filtered = GenerationBundle(
        instance_id=bundle.instance_id,
        documentation=bundle.documentation,
        facts=tuple(fact for fact in bundle.facts if fact.fact_id in fact_ids),
        questions=records,
        probes={
            path: script for path, script in bundle.probes.items() if path in probe_paths
        },
        golden_proofs={
            path: script
            for path, script in bundle.golden_proofs.items()
            if path in golden_proof_paths
        },
        metadata={**bundle.metadata, **summary},
        debug={**bundle.debug, "filtering": summary},
        schema_version=bundle.schema_version,
    )
    write_qa_bundle(filtered, output_dir)
    if bundle.schema_version == QA_SCHEMA_V2:
        golden_report_path = bundle_dir / "golden-proof-executions.json"
        if not golden_report_path.is_file():
            raise ValueError(
                "Filtered v2 bundle requires trusted golden proof executions"
            )
        golden_executions = load_proof_executions(golden_report_path)
        retained_executions = []
        for record in records:
            key = (record.question_id, "golden")
            execution = golden_executions.get(key)
            if execution is None:
                raise ValueError(
                    f"Golden proof execution is missing for {record.question_id}"
                )
            if execution.source_sha256 != record.golden_proof.sha256:
                raise ValueError(
                    f"Golden proof execution hash differs for {record.question_id}"
                )
            retained_executions.append(execution)
        write_json(
            output_dir / "golden-proof-executions.json",
            {
                "protocol": "qa_proof_execution_report/v1",
                "source": str(golden_report_path),
                "stage": "golden",
                "n_questions": len(records),
                "n_executions": len(retained_executions),
                "executions": [
                    execution.to_dict()
                    for execution in retained_executions
                ],
            },
        )
    return summary
