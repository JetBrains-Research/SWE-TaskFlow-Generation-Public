from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from interactive_benchmark_generation.dataset_io import write_json
from interactive_benchmark_generation.qa_answer_matching import match_answer
from interactive_benchmark_generation.qa_io import load_qa_records
from interactive_benchmark_generation.qa_models import (
    QA_SCHEMA_V2,
    AnswerJudgeAssessment,
    ProofExecutionRecord,
    ProofJudgeAssessment,
    QARecord,
    normalize_answer,
)
from interactive_benchmark_generation.qa_validation import (
    load_judge_assessments,
    load_proof_executions,
)


def _trajectory_answer_records(value: dict[str, Any], records: list[Any]) -> list[dict[str, Any]]:
    enriched = [dict(record) for record in records if isinstance(record, dict)]
    messages = value.get("messages")
    if not isinstance(messages, list):
        return enriched
    current_index = -1
    metrics = [
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cost": 0,
            "steps": 0,
            "tool_calls": 0,
        }
        for _ in enriched
    ]
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            current_index += 1
            continue
        if not 0 <= current_index < len(metrics):
            continue
        extra = message.get("extra")
        if not isinstance(extra, dict):
            continue
        response = extra.get("response")
        if not isinstance(response, dict):
            continue
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt_details = usage.get("prompt_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        current = metrics[current_index]
        current["input_tokens"] += usage.get("prompt_tokens", 0) or 0
        current["output_tokens"] += usage.get("completion_tokens", 0) or 0
        current["cached_tokens"] += prompt_details.get("cached_tokens", 0) or 0
        current["cost"] += extra.get("cost", 0) or 0
        current["steps"] += 1
        actions = extra.get("actions")
        if isinstance(actions, list):
            current["tool_calls"] += len(actions)
    for index, record in enumerate(enriched):
        record["metrics"] = metrics[index]
    return enriched


def _normalize_question_record(value: dict[str, Any]) -> dict[str, Any]:
    record = dict(value)
    if "answer" not in record and "frozen_answer" in record:
        record["answer"] = record["frozen_answer"]
    proof = record.get("proof")
    proof_value = dict(proof) if isinstance(proof, dict) else {}
    for source, target in (
        ("proof_script", "script"),
        ("proof_path", "path"),
        ("proof_sha256", "sha256"),
        ("proof_status", "status"),
    ):
        if source in record and target not in proof_value:
            proof_value[target] = record[source]
    if proof_value:
        record["proof"] = proof_value
    phases = record.get("phases")
    if isinstance(phases, dict):
        record.setdefault("answer_phase", phases.get("answer"))
        record.setdefault("proof_phase", phases.get("proof"))
    return record


def _materialize_sidecar_proof(root: Path, value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("proof")
    proof = dict(nested) if isinstance(nested, dict) else dict(value)
    for name in ("question_id", "question_index"):
        if name in value:
            proof.setdefault(name, value[name])
    path_value = proof.get("path", proof.get("proof_path"))
    digest = proof.get(
        "sha256",
        proof.get("hash", proof.get("source_sha256", proof.get("proof_sha256"))),
    )
    if not isinstance(path_value, str) or not path_value:
        return {**proof, "status": "malformed", "artifact_error": "missing_proof_path"}
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        return {**proof, "status": "malformed", "artifact_error": "unsafe_proof_path"}
    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root not in target.parents or not target.is_file():
        return {**proof, "status": "malformed", "artifact_error": "missing_proof_file"}
    try:
        script = target.read_text()
    except (OSError, UnicodeError):
        return {**proof, "status": "malformed", "artifact_error": "unreadable_proof_file"}
    actual_digest = hashlib.sha256(script.encode()).hexdigest()
    if not isinstance(digest, str):
        return {
            **proof,
            "status": "malformed",
            "artifact_error": "missing_proof_hash",
            "actual_sha256": actual_digest,
        }
    if actual_digest != digest:
        return {
            **proof,
            "status": "malformed",
            "artifact_error": "proof_hash_mismatch",
            "actual_sha256": actual_digest,
        }
    return {
        **proof,
        "path": path_value,
        "sha256": digest,
        "script": script,
    }


def load_agent_results(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        answers_path = path / "answers.jsonl"
        proofs_path = path / "proofs.jsonl"
        if not answers_path.is_file():
            raise ValueError(f"Missing answer sidecar: {answers_path}")
        answers = load_agent_results(answers_path)
        proofs = load_agent_results(proofs_path) if proofs_path.is_file() else []
        proofs_by_key = {}
        for proof in proofs:
            proof = _materialize_sidecar_proof(path, proof)
            key = proof.get("question_id", proof.get("question_index"))
            if key in proofs_by_key:
                raise ValueError(f"Duplicate proof sidecar record: {key}")
            proofs_by_key[key] = proof
        merged = []
        for answer in answers:
            key = answer.get("question_id", answer.get("question_index"))
            proof = proofs_by_key.pop(key, None)
            merged.append(
                _normalize_question_record(
                    {**answer, **({"proof": proof} if proof is not None else {})}
                )
            )
        merged.extend({"proof": proof, **proof} for proof in proofs_by_key.values())
        return merged
    try:
        text = path.read_text()
    except OSError as error:
        raise ValueError(f"Could not read agent results {path}: {error}") from error
    if path.suffix == ".jsonl":
        values = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            values.append(value)
        return values
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        info = value.get("info")
        if isinstance(info, dict) and isinstance(info.get("question_records"), list):
            records = [
                _normalize_question_record(record)
                for record in info["question_records"]
                if isinstance(record, dict)
            ]
        elif isinstance(info, dict) and isinstance(info.get("answer_records"), list):
            records = _trajectory_answer_records(value, info["answer_records"])
        else:
            records = value.get("results", value.get("records", value.get("answers")))
        if records is None and "question_id" in value:
            records = [value]
    else:
        records = None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Agent results must contain a list of result objects")
    return records


def _answer_from_result(result: dict[str, Any]) -> Any:
    for field in ("answer", "submitted_answer", "submission", "model_answer", "response"):
        if field in result:
            return result[field]
    return None


def _number(result: dict[str, Any], *fields: str) -> int | float:
    usage = result.get("usage", {})
    metrics = result.get("metrics", {})
    sources = [result]
    if isinstance(usage, dict):
        sources.append(usage)
    if isinstance(metrics, dict):
        sources.append(metrics)
    for source in sources:
        for field in fields:
            value = source.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
    return 0


def _phase_metrics(result: dict[str, Any], phase: str) -> dict[str, int | float]:
    value = result.get(f"{phase}_metrics")
    if not isinstance(value, dict):
        value = result.get(f"{phase}_phase")
    if not isinstance(value, dict) and phase == "proof":
        proof = result.get("proof")
        if isinstance(proof, dict):
            value = proof.get("proof_phase")
    if isinstance(value, dict) and isinstance(value.get("metrics"), dict):
        value = value["metrics"]
    if not isinstance(value, dict):
        value = result.get("metrics", {}) if phase == "answer" else {}
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
    }
    usage_records = value.get("usage_records")
    if isinstance(usage_records, list):
        for record in usage_records:
            if not isinstance(record, dict):
                continue
            usage = record.get("usage")
            if not isinstance(usage, dict):
                continue
            prompt_details = usage.get("prompt_tokens_details")
            if not isinstance(prompt_details, dict):
                prompt_details = {}
            usage_totals["input_tokens"] += usage.get("prompt_tokens", 0) or 0
            usage_totals["output_tokens"] += usage.get("completion_tokens", 0) or 0
            usage_totals["cached_tokens"] += prompt_details.get("cached_tokens", 0) or 0
    return {
        "input_tokens": (
            _number(value, "input_tokens", "prompt_tokens")
            or usage_totals["input_tokens"]
        ),
        "output_tokens": (
            _number(value, "output_tokens", "completion_tokens")
            or usage_totals["output_tokens"]
        ),
        "cached_tokens": (
            _number(value, "cached_tokens", "cache_read_input_tokens")
            or usage_totals["cached_tokens"]
        ),
        "cost": _number(value, "cost", "instance_cost"),
        "steps": _number(value, "steps", "n_steps"),
        "tool_calls": _number(value, "tool_calls", "n_tool_calls"),
        "latency_seconds": _number(
            value,
            "latency_seconds",
            "latency",
            "duration_seconds",
        ),
    }


def _proof_submission(result: dict[str, Any]) -> dict[str, Any]:
    proof = result.get("proof")
    proof_value = proof if isinstance(proof, dict) else {}
    script = result.get("proof_script", proof_value.get("script"))
    path = result.get("proof_path", proof_value.get("path"))
    digest = result.get("proof_sha256", proof_value.get("sha256", proof_value.get("hash")))
    status = result.get("proof_status", proof_value.get("status"))
    artifact_error = proof_value.get("artifact_error")
    if isinstance(script, str) and not isinstance(digest, str):
        digest = hashlib.sha256(script.encode()).hexdigest()
    submitted = artifact_error is None and (
        isinstance(script, str)
        or isinstance(path, str)
        or status in {"submitted", "captured", "executed", "completed"}
    )
    return {
        "submitted": submitted,
        "script": script if isinstance(script, str) else None,
        "path": path if isinstance(path, str) else None,
        "sha256": digest if isinstance(digest, str) else None,
        "status": status,
        "artifact_error": artifact_error,
    }


def _metric_totals(values: list[dict[str, int | float]]) -> dict[str, int | float]:
    fields = (
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "cost",
        "steps",
        "tool_calls",
        "latency_seconds",
    )
    return {field: sum(value[field] for value in values) for field in fields}


def _evaluate_proof_question(
    record: QARecord,
    attempts: list[dict[str, Any]],
    executions: dict[tuple[str, str], ProofExecutionRecord],
    assessments: dict[str, ProofJudgeAssessment],
) -> dict[str, Any]:
    result = attempts[-1] if attempts else {}
    answer = _answer_from_result(result) if attempts else None
    proof = _proof_submission(result)
    golden_execution = executions.get((record.question_id, "golden"))
    agent_execution = executions.get((record.question_id, "agent"))
    assessment = assessments.get(record.question_id)
    hash_matches = (
        agent_execution is not None
        and proof["sha256"] is not None
        and agent_execution.source_sha256 == proof["sha256"]
    )
    if agent_execution is not None and proof["sha256"] is None:
        hash_matches = proof["submitted"]
    dataset_issue = assessment.dataset_issue if assessment is not None else False
    answer_correct = assessment.answer_correct if assessment is not None else False
    answer_plus_proof_correct = (
        assessment.answer_plus_proof_correct(agent_execution)
        if assessment is not None and hash_matches and proof["submitted"]
        else False
    )
    failures = (
        [code.value for code in assessment.failure_codes]
        if assessment is not None
        else ["missing_judge_assessment"]
    )
    if not proof["submitted"]:
        failures.append("missing_agent_proof")
        if proof["artifact_error"] is not None:
            failures.append(proof["artifact_error"])
    elif agent_execution is None:
        failures.append("missing_agent_proof_execution")
    elif not agent_execution.success:
        failures.append("agent_proof_execution_failed")
    if agent_execution is not None and not hash_matches:
        failures.append("agent_proof_hash_mismatch")
    return {
        "instance_id": record.instance_id,
        "question_id": record.question_id,
        "question": record.question,
        "golden_answer": record.golden_answer.to_dict(),
        "golden_proof": record.golden_proof.to_dict(),
        "answer": answer,
        "missing": not attempts or answer is None,
        "runtime_error": result.get("error") if attempts else None,
        "retries": max(0, len(attempts) - 1),
        "proof": proof,
        "golden_proof_execution": (
            golden_execution.to_dict() if golden_execution is not None else None
        ),
        "agent_proof_execution": (
            agent_execution.to_dict() if agent_execution is not None else None
        ),
        "judge_assessment": assessment.to_dict() if assessment is not None else None,
        "dataset_issue": dataset_issue,
        "answer_correct": answer_correct,
        "answer_plus_proof_correct": answer_plus_proof_correct,
        "proof_submitted": proof["submitted"],
        "proof_executed": agent_execution.executed if agent_execution is not None else False,
        "proof_execution_success": (
            agent_execution.success if agent_execution is not None else False
        ),
        "proof_matches_answer": (
            assessment.agent_proof_matches_answer.value == "yes"
            if assessment is not None
            else False
        ),
        "proof_establishes_answer": (
            assessment.agent_proof_establishes_answer.value == "yes"
            if assessment is not None
            else False
        ),
        "valid_alternative": (
            assessment.valid_alternative_to_golden
            if assessment is not None
            else False
        ),
        "failures": list(dict.fromkeys(failures)),
        "answer_metrics": _phase_metrics(result, "answer"),
        "proof_metrics": _phase_metrics(result, "proof"),
    }


def _proof_report(
    records: list[QARecord],
    by_question: dict[str, list[dict[str, Any]]],
    unknown_results: list[dict[str, Any]],
    executions: dict[tuple[str, str], ProofExecutionRecord],
    assessments: dict[str, ProofJudgeAssessment],
    judge_metadata: dict[str, Any],
) -> dict[str, Any]:
    questions = [
        _evaluate_proof_question(
            record,
            by_question.get(record.question_id, []),
            executions,
            assessments,
        )
        for record in records
    ]
    scored = [question for question in questions if not question["dataset_issue"]]
    denominator = len(scored)
    n_answer_correct = sum(question["answer_correct"] for question in scored)
    n_answer_plus_proof_correct = sum(
        question["answer_plus_proof_correct"] for question in scored
    )
    answer_metrics = _metric_totals([question["answer_metrics"] for question in questions])
    proof_metrics = _metric_totals([question["proof_metrics"] for question in questions])
    average_answer_steps = answer_metrics["steps"] / len(questions) if questions else 0.0
    average_proof_steps = proof_metrics["steps"] / len(questions) if questions else 0.0
    judge_metrics = judge_metadata.get("metrics", {})
    if not isinstance(judge_metrics, dict):
        judge_metrics = {}
    judge_cost = judge_metadata.get("cost", 0)
    if not isinstance(judge_cost, (int, float)) or isinstance(judge_cost, bool):
        judge_cost = 0
    failure_counts: dict[str, int] = {}
    for question in questions:
        for failure in question["failures"]:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    def rate(field: str) -> float:
        return (
            sum(question[field] for question in scored) / denominator
            if denominator
            else 0.0
        )
    return {
        "schema_version": QA_SCHEMA_V2,
        "evaluation_mode": "proof_judged_v1",
        "n_questions": len(questions),
        "n_scored": denominator,
        "n_dataset_issues": len(questions) - denominator,
        "dataset_issue_rate": (
            (len(questions) - denominator) / len(questions) if questions else 0.0
        ),
        "n_answer_correct": n_answer_correct,
        "answer_accuracy": (
            n_answer_correct / denominator if denominator else 0.0
        ),
        "n_answer_plus_proof_correct": n_answer_plus_proof_correct,
        "answer_plus_proof_accuracy": (
            n_answer_plus_proof_correct / denominator if denominator else 0.0
        ),
        "proof_submission_rate": rate("proof_submitted"),
        "proof_execution_rate": rate("proof_execution_success"),
        "proof_consistency_rate": rate("proof_matches_answer"),
        "proof_establishment_rate": rate("proof_establishes_answer"),
        "valid_alternative_rate": rate("valid_alternative"),
        "answer_correct_proof_failed": sum(
            question["answer_correct"] and not question["answer_plus_proof_correct"]
            for question in scored
        ),
        "answer_wrong_proof_apparently_valid": sum(
            not question["answer_correct"] and question["proof_establishes_answer"]
            for question in scored
        ),
        "phase_metrics": {
            "answer": answer_metrics,
            "proof": proof_metrics,
            "judge": {**judge_metrics, "cost": judge_cost},
        },
        "average_answer_steps": average_answer_steps,
        "average_proof_steps": average_proof_steps,
        "average_steps_per_answer": average_answer_steps + average_proof_steps,
        "judge_cost": judge_cost,
        "failure_counts": failure_counts,
        "unknown_results": unknown_results,
        "questions": questions,
    }


def _answer_report(
    records: list[QARecord],
    by_question: dict[str, list[dict[str, Any]]],
    unknown_results: list[dict[str, Any]],
    executions: dict[tuple[str, str], ProofExecutionRecord],
    assessments: dict[str, AnswerJudgeAssessment],
    judge_metadata: dict[str, Any],
) -> dict[str, Any]:
    questions = []
    for record in records:
        attempts = by_question.get(record.question_id, [])
        result = attempts[-1] if attempts else {}
        answer = _answer_from_result(result) if attempts else None
        execution = executions.get((record.question_id, "golden"))
        assessment = assessments.get(record.question_id)
        failures = (
            [code.value for code in assessment.failure_codes]
            if assessment is not None
            else ["missing_judge_assessment"]
        )
        questions.append(
            {
                "instance_id": record.instance_id,
                "question_id": record.question_id,
                "question": record.question,
                "golden_answer": record.golden_answer.to_dict(),
                "golden_proof": record.golden_proof.to_dict(),
                "answer": answer,
                "missing": not attempts or answer is None,
                "runtime_error": result.get("error") if attempts else None,
                "retries": max(0, len(attempts) - 1),
                "golden_proof_execution": (
                    execution.to_dict() if execution is not None else None
                ),
                "judge_assessment": (
                    assessment.to_dict() if assessment is not None else None
                ),
                "dataset_issue": (
                    assessment.dataset_issue
                    if assessment is not None
                    else False
                ),
                "answer_correct": (
                    assessment.answer_correct
                    if assessment is not None
                    else False
                ),
                "answer_plus_proof_correct": None,
                "proof_submitted": None,
                "proof_executed": None,
                "proof_execution_success": None,
                "proof_matches_answer": None,
                "proof_establishes_answer": None,
                "valid_alternative": (
                    assessment.valid_alternative_to_golden
                    if assessment is not None
                    else False
                ),
                "failures": list(dict.fromkeys(failures)),
                "answer_metrics": _phase_metrics(result, "answer"),
                "proof_metrics": _phase_metrics(result, "proof"),
            }
        )
    scored = [
        question for question in questions if not question["dataset_issue"]
    ]
    denominator = len(scored)
    n_answer_correct = sum(
        question["answer_correct"] for question in scored
    )
    answer_metrics = _metric_totals(
        [question["answer_metrics"] for question in questions]
    )
    proof_metrics = _metric_totals(
        [question["proof_metrics"] for question in questions]
    )
    if any(proof_metrics.values()):
        raise ValueError("Answer-only results contain proof-phase metrics")
    judge_metrics = judge_metadata.get("metrics", {})
    if not isinstance(judge_metrics, dict):
        judge_metrics = {}
    judge_cost = judge_metadata.get("cost", 0)
    if not isinstance(judge_cost, (int, float)) or isinstance(judge_cost, bool):
        judge_cost = 0
    failure_counts: dict[str, int] = {}
    for question in questions:
        for failure in question["failures"]:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    return {
        "schema_version": QA_SCHEMA_V2,
        "evaluation_mode": "answer_judged_v1",
        "scoring_mode": "answer_only",
        "n_questions": len(questions),
        "n_scored": denominator,
        "n_dataset_issues": len(questions) - denominator,
        "dataset_issue_rate": (
            (len(questions) - denominator) / len(questions)
            if questions
            else 0.0
        ),
        "n_answer_correct": n_answer_correct,
        "answer_accuracy": (
            n_answer_correct / denominator if denominator else 0.0
        ),
        "n_answer_plus_proof_correct": None,
        "answer_plus_proof_accuracy": None,
        "proof_submission_rate": None,
        "proof_execution_rate": None,
        "proof_consistency_rate": None,
        "proof_establishment_rate": None,
        "valid_alternative_rate": (
            sum(question["valid_alternative"] for question in scored)
            / denominator
            if denominator
            else 0.0
        ),
        "answer_correct_proof_failed": None,
        "answer_wrong_proof_apparently_valid": None,
        "phase_metrics": {
            "answer": answer_metrics,
            "proof": proof_metrics,
            "judge": {**judge_metrics, "cost": judge_cost},
        },
        "average_answer_steps": (
            answer_metrics["steps"] / len(questions) if questions else 0.0
        ),
        "average_proof_steps": 0.0,
        "average_steps_per_answer": (
            answer_metrics["steps"] / len(questions) if questions else 0.0
        ),
        "judge_cost": judge_cost,
        "failure_counts": failure_counts,
        "unknown_results": unknown_results,
        "questions": questions,
    }


def _evaluate_question(record: QARecord, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    result = attempts[-1] if attempts else {}
    answer = _answer_from_result(result) if attempts else None
    malformed = answer is not None and not isinstance(answer, str)
    normalized_answer = normalize_answer(answer) if isinstance(answer, str) else None
    normalized_accepted = [normalize_answer(value) for value in record.accepted_answers]
    runtime_error = result.get("error") if attempts else None
    answer_match = match_answer(record, answer) if isinstance(answer, str) else None
    if answer_match is not None:
        malformed = malformed or answer_match.malformed
    correct = (
        answer_match is not None
        and answer_match.matched
        and not runtime_error
    )
    return {
        "instance_id": record.instance_id,
        "question_id": record.question_id,
        "question": record.question,
        "accepted_answers": list(record.accepted_answers),
        "normalized_accepted_answers": normalized_accepted,
        "answer": answer,
        "normalized_answer": normalized_answer,
        "answer_contract": (
            record.answer_contract.to_dict() if record.answer_contract is not None else None
        ),
        "match_method": answer_match.method if answer_match is not None else None,
        "match_reason": answer_match.reason if answer_match is not None else None,
        "mismatch_path": answer_match.mismatch_path if answer_match is not None else None,
        "correct": correct,
        "missing": not attempts or answer is None,
        "malformed": malformed,
        "retries": max(0, len(attempts) - 1),
        "runtime_error": runtime_error,
        "input_tokens": _number(result, "input_tokens", "prompt_tokens"),
        "output_tokens": _number(result, "output_tokens", "completion_tokens"),
        "cached_tokens": _number(result, "cached_tokens", "cache_read_input_tokens"),
        "cost": _number(result, "cost", "instance_cost"),
        "steps": _number(result, "steps", "n_steps"),
        "tool_calls": _number(result, "tool_calls", "n_tool_calls"),
        "latency_seconds": _number(result, "latency_seconds", "latency", "duration_seconds"),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# QA Evaluation",
        "",
        f"Correct: {report['n_correct']}/{report['n_questions']} ({report['correct_proportion']:.1%})",
        "",
        f"Input tokens: {report['totals']['input_tokens']}",
        "",
        f"Output tokens: {report['totals']['output_tokens']}",
        "",
        f"Cached tokens: {report['totals']['cached_tokens']}",
        "",
        f"Cost: {report['totals']['cost']}",
        "",
        f"Average steps per answer: {report['average_steps_per_answer']:.2f}",
        "",
        f"Latency: {report['totals']['latency_seconds']} seconds",
    ]
    for result in report["questions"]:
        lines.extend(
            [
                "",
                f"## {result['question_id']}",
                "",
                result["question"],
                "",
                f"Accepted answers: {', '.join(result['accepted_answers'])}",
                "",
                f"Agent answer: {result['answer'] if result['answer'] is not None else 'MISSING'}",
                "",
                f"Normalized accepted: {', '.join(result['normalized_accepted_answers'])}",
                "",
                f"Normalized answer: {result['normalized_answer'] if result['normalized_answer'] is not None else 'MISSING'}",
                "",
                f"Match method: {result['match_method'] if result['match_method'] is not None else 'NONE'}",
                "",
                f"Match reason: {result['match_reason'] if result['match_reason'] is not None else 'NONE'}",
                "",
                f"Mismatch path: {result['mismatch_path'] if result['mismatch_path'] is not None else 'NONE'}",
                "",
                f"Steps: {result['steps']}",
                "",
                f"Result: {'CORRECT' if result['correct'] else 'INCORRECT'}",
            ]
        )
        if result["runtime_error"]:
            lines.extend(["", f"Runtime error: {result['runtime_error']}"])
        if result["retries"]:
            lines.extend(["", f"Retries: {result['retries']}"])
    return "\n".join(lines) + "\n"


def evaluate_agent_results(
    bundle_dir: Path,
    agent_results_path: Path,
    output_dir: Path,
    proof_executions_path: Path | None = None,
    judge_report_path: Path | None = None,
    judge_cost_override: float | None = None,
) -> dict[str, Any]:
    records = load_qa_records(bundle_dir)
    agent_results = load_agent_results(agent_results_path)
    by_question: dict[str, list[dict[str, Any]]] = {}
    unknown_results = []
    known_ids = {record.question_id for record in records}
    for result in agent_results:
        question_id = result.get("question_id")
        question_index = result.get("question_index")
        if question_id is None and isinstance(question_index, int) and 0 <= question_index < len(records):
            question_id = records[question_index].question_id
        if question_id not in known_ids:
            unknown_results.append(result)
            continue
        by_question.setdefault(question_id, []).append(result)
    proof_enabled = bool(records) and all(
        record.schema_version == QA_SCHEMA_V2 for record in records
    )
    if proof_enabled:
        execution_path = proof_executions_path or bundle_dir / "proof-executions.json"
        judge_path = judge_report_path
        if judge_path is None:
            answer_judge_path = bundle_dir / "answer-judge-assessments.json"
            judge_path = (
                answer_judge_path
                if answer_judge_path.is_file()
                else bundle_dir / "judge-assessments.json"
            )
        executions = (
            load_proof_executions(execution_path) if execution_path.is_file() else {}
        )
        if judge_path.is_file():
            assessments, judge_metadata = load_judge_assessments(judge_path)
        else:
            assessments, judge_metadata = {}, {}
        if judge_cost_override is not None:
            judge_metadata = {**judge_metadata, "cost": judge_cost_override}
        if judge_metadata.get("protocol") == "qa_answer_judge_report/v1":
            report = _answer_report(
                records,
                by_question,
                unknown_results,
                executions,
                assessments,
                judge_metadata,
            )
        else:
            report = _proof_report(
                records,
                by_question,
                unknown_results,
                executions,
                assessments,
                judge_metadata,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "evaluation.json", report)
        answer_only = report["evaluation_mode"] == "answer_judged_v1"
        lines = [
            "# QA Answer Evaluation"
            if answer_only
            else "# QA Proof Evaluation",
            "",
            f"Answer accuracy: {report['n_answer_correct']}/{report['n_scored']} "
            f"({report['answer_accuracy']:.1%})",
            "",
            f"Dataset issues: {report['n_dataset_issues']}/{report['n_questions']}",
            "",
            f"Valid alternative rate: {report['valid_alternative_rate']:.1%}",
            "",
            f"Average steps per answer: {report['average_steps_per_answer']:.2f}",
            "",
            f"Average answer-phase steps: {report['average_answer_steps']:.2f}",
            "",
            f"Answer-phase cost: {report['phase_metrics']['answer']['cost']}",
            "",
            f"Judge cost: {report['judge_cost']}",
        ]
        if not answer_only:
            lines[6:6] = [
                f"Answer plus proof accuracy: "
                f"{report['n_answer_plus_proof_correct']}/{report['n_scored']} "
                f"({report['answer_plus_proof_accuracy']:.1%})",
                "",
                f"Proof submission rate: "
                f"{report['proof_submission_rate']:.1%}",
                "",
                f"Proof execution rate: "
                f"{report['proof_execution_rate']:.1%}",
                "",
                f"Proof consistency rate: "
                f"{report['proof_consistency_rate']:.1%}",
                "",
                f"Proof establishment rate: "
                f"{report['proof_establishment_rate']:.1%}",
                "",
                f"Average proof-phase steps: "
                f"{report['average_proof_steps']:.2f}",
                "",
                f"Proof-phase cost: "
                f"{report['phase_metrics']['proof']['cost']}",
                "",
            ]
        for question in report["questions"]:
            lines.extend(
                [
                    "",
                    f"## {question['question_id']}",
                    "",
                    question["question"],
                    "",
                    f"Golden reference: {question['golden_answer']['canonical']}",
                    "",
                    f"Agent answer: {question['answer'] if question['answer'] is not None else 'MISSING'}",
                    "",
                    f"Answer result: {'CORRECT' if question['answer_correct'] else 'INCORRECT'}",
                    "",
                    f"Dataset issue: {question['dataset_issue']}",
                    "",
                    f"Failures: {', '.join(question['failures']) or 'NONE'}",
                ]
            )
            if not answer_only:
                lines.extend(
                    [
                        "",
                        f"Answer plus proof result: "
                        f"{'CORRECT' if question['answer_plus_proof_correct'] else 'INCORRECT'}",
                    ]
                )
        (output_dir / "evaluation.md").write_text("\n".join(lines) + "\n")
        return report
    if any(record.schema_version == QA_SCHEMA_V2 for record in records):
        raise ValueError("Cannot evaluate mixed v1 and v2 records")
    questions = [_evaluate_question(record, by_question.get(record.question_id, [])) for record in records]
    n_correct = sum(result["correct"] for result in questions)
    totals = {
        field: sum(result[field] for result in questions)
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "cost",
            "steps",
            "tool_calls",
            "latency_seconds",
        )
    }
    report = {
        "n_questions": len(questions),
        "n_correct": n_correct,
        "correct_proportion": n_correct / len(questions) if questions else 0.0,
        "average_steps_per_answer": totals["steps"] / len(questions) if questions else 0.0,
        "totals": totals,
        "unknown_results": unknown_results,
        "questions": questions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation.json", report)
    (output_dir / "evaluation.md").write_text(_markdown_report(report))
    return report


def evaluate_agent_run(
    bundle_root: Path,
    agent_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    instances_dir = bundle_root / "instances"
    bundle_dirs = {
        path.name: path
        for path in instances_dir.iterdir()
        if path.is_dir() and (path / "questions.jsonl").is_file()
    }
    trajectories = {}
    for trajectory in agent_run_dir.glob("*/agent/mini-swe-agent.trajectory.json"):
        result_path = trajectory.parents[1] / "result.json"
        result = json.loads(result_path.read_text())
        task_name = result.get("task_name")
        if not isinstance(task_name, str):
            raise ValueError(f"Missing task_name in {result_path}")
        trajectories[task_name.removeprefix("cais/")] = trajectory
    if set(trajectories) != set(bundle_dirs):
        raise ValueError(
            f"Agent run instances differ: missing={sorted(set(bundle_dirs) - set(trajectories))}, "
            f"unknown={sorted(set(trajectories) - set(bundle_dirs))}"
        )
    instance_reports = []
    for instance_id in sorted(bundle_dirs):
        report = evaluate_agent_results(
            bundle_dirs[instance_id],
            trajectories[instance_id],
            output_dir / "instances" / instance_id,
        )
        instance_reports.append({"instance_id": instance_id, **report})
    questions = [
        question
        for instance_report in instance_reports
        for question in instance_report["questions"]
    ]
    proof_enabled = bool(instance_reports) and all(
        report.get("schema_version") == QA_SCHEMA_V2 for report in instance_reports
    )
    if proof_enabled:
        n_scored = sum(report["n_scored"] for report in instance_reports)
        n_answer_correct = sum(
            report["n_answer_correct"] for report in instance_reports
        )
        n_answer_plus_proof_correct = sum(
            report["n_answer_plus_proof_correct"] for report in instance_reports
        )
        n_questions = len(questions)
        report = {
            "schema_version": QA_SCHEMA_V2,
            "evaluation_mode": "proof_judged_v1",
            "n_instances": len(instance_reports),
            "n_questions": n_questions,
            "n_scored": n_scored,
            "n_dataset_issues": n_questions - n_scored,
            "dataset_issue_rate": (
                (n_questions - n_scored) / n_questions if n_questions else 0.0
            ),
            "n_answer_correct": n_answer_correct,
            "answer_accuracy": (
                n_answer_correct / n_scored if n_scored else 0.0
            ),
            "n_answer_plus_proof_correct": n_answer_plus_proof_correct,
            "answer_plus_proof_accuracy": (
                n_answer_plus_proof_correct / n_scored if n_scored else 0.0
            ),
            "judge_cost": sum(
                instance_report["judge_cost"]
                for instance_report in instance_reports
            ),
            "instances": instance_reports,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "evaluation.json", report)
        lines = [
            "# QA Proof Batch Evaluation",
            "",
            f"Instances: {report['n_instances']}",
            "",
            f"Answer accuracy: {n_answer_correct}/{n_scored} "
            f"({report['answer_accuracy']:.1%})",
            "",
            f"Answer plus proof accuracy: {n_answer_plus_proof_correct}/{n_scored} "
            f"({report['answer_plus_proof_accuracy']:.1%})",
            "",
            f"Dataset issues: {report['n_dataset_issues']}/{n_questions}",
            "",
            f"Judge cost: {report['judge_cost']}",
        ]
        (output_dir / "evaluation.md").write_text("\n".join(lines) + "\n")
        return report
    if any(report.get("schema_version") == QA_SCHEMA_V2 for report in instance_reports):
        raise ValueError("Cannot aggregate mixed v1 and v2 instance reports")
    totals = {
        field: sum(instance_report["totals"][field] for instance_report in instance_reports)
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "cost",
            "steps",
            "tool_calls",
            "latency_seconds",
        )
    }
    n_correct = sum(question["correct"] for question in questions)
    report = {
        "n_instances": len(instance_reports),
        "n_questions": len(questions),
        "n_correct": n_correct,
        "correct_proportion": n_correct / len(questions) if questions else 0.0,
        "average_steps_per_answer": totals["steps"] / len(questions) if questions else 0.0,
        "totals": totals,
        "instances": instance_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation.json", report)
    lines = [
        "# QA Batch Evaluation",
        "",
        f"Instances: {report['n_instances']}",
        "",
        f"Correct: {n_correct}/{len(questions)} ({report['correct_proportion']:.1%})",
        "",
        f"Average steps per answer: {report['average_steps_per_answer']:.2f}",
        "",
        f"Cost: {totals['cost']}",
    ]
    for instance_report in instance_reports:
        lines.extend(
            [
                "",
                f"## {instance_report['instance_id']}",
                "",
                f"Correct: {instance_report['n_correct']}/{instance_report['n_questions']} "
                f"({instance_report['correct_proportion']:.1%})",
                "",
                f"Average steps per answer: "
                f"{instance_report['average_steps_per_answer']:.2f}",
            ]
        )
        for question in instance_report["questions"]:
            lines.extend(
                [
                    "",
                    f"### {question['question_id']}",
                    "",
                    question["question"],
                    "",
                    f"Accepted: {', '.join(question['accepted_answers'])}",
                    "",
                    f"Agent: {question['answer'] if question['answer'] is not None else 'MISSING'}",
                    "",
                    f"Match: {question['match_method'] if question['match_method'] is not None else 'NONE'}",
                    "",
                    f"Match reason: {question['match_reason'] if question['match_reason'] is not None else 'NONE'}",
                    "",
                    f"Mismatch path: {question['mismatch_path'] if question['mismatch_path'] is not None else 'NONE'}",
                    "",
                    f"Steps: {question['steps']}",
                    "",
                    f"Result: {'CORRECT' if question['correct'] else 'INCORRECT'}",
                ]
            )
    (output_dir / "evaluation.md").write_text("\n".join(lines) + "\n")
    return report
