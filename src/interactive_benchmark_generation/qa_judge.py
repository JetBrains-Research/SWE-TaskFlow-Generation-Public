from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from interactive_benchmark_generation.dataset_io import write_json
from interactive_benchmark_generation.llm import call_json_model
from interactive_benchmark_generation.qa_evaluation import (
    _answer_from_result,
    _proof_submission,
    load_agent_results,
)
from interactive_benchmark_generation.qa_io import load_generation_bundle
from interactive_benchmark_generation.qa_judge_prompts import (
    QA_ANSWER_JUDGE_SYSTEM_PROMPT,
    QA_JUDGE_SYSTEM_PROMPT,
    build_qa_answer_judge_prompt,
    build_qa_judge_prompt,
)
from interactive_benchmark_generation.qa_models import (
    AnswerJudgeAssessment,
    JudgeFailureCode,
    ProofExecutionRecord,
    ProofJudgeAssessment,
    QARecord,
)
from interactive_benchmark_generation.qa_validation import load_proof_executions

ANSWER_JUDGE_FAILURE_CODES = [
    JudgeFailureCode.MISSING_AGENT_ANSWER,
    JudgeFailureCode.AGENT_ANSWER_INCORRECT,
    JudgeFailureCode.GOLDEN_PROOF_EXECUTION_FAILED,
    JudgeFailureCode.GOLDEN_PROOF_UNSUPPORTED,
    JudgeFailureCode.GOLD_REFERENCE_INCOMPLETE,
    JudgeFailureCode.GOLD_REFERENCE_CONTRADICTED,
    JudgeFailureCode.JUDGE_INCONCLUSIVE,
    JudgeFailureCode.INVALID_EVIDENCE,
]


def judge_proof_submission(
    record: QARecord,
    golden_proof_source: str,
    golden_execution: ProofExecutionRecord,
    agent_answer: str,
    agent_proof_source: str,
    agent_execution: ProofExecutionRecord,
    model: str = "openai/gpt-5.6-sol",
    reasoning_effort: str = "medium",
    max_attempts: int = 3,
    request_timeout: float | None = 90.0,
    retry_delay: float = 1.0,
    metrics: dict[str, Any] | None = None,
) -> ProofJudgeAssessment:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    prompt = build_qa_judge_prompt(
        question=record.question,
        golden_answer=record.golden_answer.canonical,
        golden_proof_source=golden_proof_source,
        golden_execution=golden_execution.to_dict(),
        agent_answer=agent_answer,
        agent_proof_source=agent_proof_source,
        agent_execution=agent_execution.to_dict(),
        question_id=record.question_id,
        failure_codes=[code.value for code in JudgeFailureCode],
    )
    last_error = None
    total_cost = 0.0
    total_usage: dict[str, int | float] = {}
    for attempt in range(max_attempts):
        call_metadata: dict[str, Any] = {}
        try:
            payload = call_json_model(
                model,
                QA_JUDGE_SYSTEM_PROMPT,
                prompt,
                request_timeout,
                reasoning_effort,
                call_metadata,
            )
            cost = call_metadata.get("cost", 0)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                total_cost += cost
            usage = call_metadata.get("usage", {})
            if isinstance(usage, dict):
                for name, value in usage.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        total_usage[name] = total_usage.get(name, 0) + value
            assessment = ProofJudgeAssessment.from_dict(payload)
            if assessment.question_id != record.question_id:
                raise ValueError("Judge response question_id differs from request")
            if metrics is not None:
                metrics.update({"cost": total_cost, **total_usage})
            return assessment
        except Exception as error:
            last_error = error
            if attempt + 1 < max_attempts and retry_delay:
                time.sleep(min(retry_delay * 2**attempt, 30.0))
    raise ValueError(
        f"Could not obtain a valid judge assessment for {record.question_id}: {last_error}"
    ) from last_error


def judge_answer_submission(
    record: QARecord,
    golden_proof_source: str,
    golden_execution: ProofExecutionRecord,
    agent_answer: str,
    checkpoint_context: dict[str, Any] | None = None,
    model: str = "openai/gpt-5.6-sol",
    reasoning_effort: str = "medium",
    max_attempts: int = 3,
    request_timeout: float | None = 90.0,
    retry_delay: float = 1.0,
    metrics: dict[str, Any] | None = None,
) -> AnswerJudgeAssessment:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    prompt = build_qa_answer_judge_prompt(
        question=record.question,
        golden_answer=record.golden_answer.canonical,
        golden_proof_source=golden_proof_source,
        golden_execution=golden_execution.to_dict(),
        agent_answer=agent_answer,
        question_id=record.question_id,
        failure_codes=[code.value for code in ANSWER_JUDGE_FAILURE_CODES],
        checkpoint_context=checkpoint_context,
    )
    last_error = None
    total_cost = 0.0
    total_usage: dict[str, int | float] = {}
    for attempt in range(max_attempts):
        call_metadata: dict[str, Any] = {}
        try:
            payload = call_json_model(
                model,
                QA_ANSWER_JUDGE_SYSTEM_PROMPT,
                prompt,
                request_timeout,
                reasoning_effort,
                call_metadata,
            )
            cost = call_metadata.get("cost", 0)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                total_cost += cost
            usage = call_metadata.get("usage", {})
            if isinstance(usage, dict):
                for name, value in usage.items():
                    if isinstance(value, (int, float)) and not isinstance(
                        value,
                        bool,
                    ):
                        total_usage[name] = total_usage.get(name, 0) + value
            assessment = AnswerJudgeAssessment.from_dict(payload)
            if assessment.question_id != record.question_id:
                raise ValueError(
                    "Answer judge response question_id differs from request"
                )
            if metrics is not None:
                metrics.update({"cost": total_cost, **total_usage})
            return assessment
        except Exception as error:
            last_error = error
            if attempt + 1 < max_attempts and retry_delay:
                time.sleep(min(retry_delay * 2**attempt, 30.0))
    raise ValueError(
        f"Could not obtain a valid answer judge assessment for "
        f"{record.question_id}: {last_error}"
    ) from last_error


def write_judge_report(
    output_path: Path,
    assessments: list[ProofJudgeAssessment],
    model: str,
    reasoning_effort: str,
    errors: list[dict[str, Any]] | None = None,
    cost: float = 0.0,
) -> dict[str, Any]:
    report = {
        "protocol": "qa_proof_judge_report/v1",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "cost": cost,
        "assessments": [assessment.to_dict() for assessment in assessments],
        "errors": errors or [],
    }
    write_json(output_path, report)
    return report


def write_answer_judge_report(
    output_path: Path,
    assessments: list[AnswerJudgeAssessment],
    model: str,
    reasoning_effort: str,
    errors: list[dict[str, Any]] | None = None,
    cost: float = 0.0,
) -> dict[str, Any]:
    report = {
        "protocol": "qa_answer_judge_report/v1",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "cost": cost,
        "assessments": [
            assessment.to_dict() for assessment in assessments
        ],
        "errors": errors or [],
    }
    write_json(output_path, report)
    return report


def judge_bundle_answers(
    bundle_dir: Path,
    agent_results_path: Path,
    golden_executions_path: Path,
    output_path: Path,
    checkpoint_context_path: Path | None = None,
    model: str = "openai/gpt-5.6-sol",
    reasoning_effort: str = "medium",
    max_attempts: int = 3,
    request_timeout: float | None = 90.0,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    bundle = load_generation_bundle(bundle_dir)
    results = load_agent_results(agent_results_path)
    executions = load_proof_executions(golden_executions_path)
    checkpoint_context = (
        json.loads(checkpoint_context_path.read_text())
        if checkpoint_context_path is not None
        else None
    )
    by_id = {}
    for result in results:
        question_id = result.get("question_id")
        question_index = result.get("question_index")
        if (
            question_id is None
            and isinstance(question_index, int)
            and 0 <= question_index < len(bundle.questions)
        ):
            question_id = bundle.questions[question_index].question_id
        if isinstance(question_id, str):
            by_id[question_id] = result
    assessments = []
    errors = []
    judge_metrics: dict[str, int | float] = {}
    for record in bundle.questions:
        result = by_id.get(record.question_id, {})
        answer = _answer_from_result(result)
        golden_execution = executions.get((record.question_id, "golden"))
        if golden_execution is None:
            errors.append(
                {
                    "question_id": record.question_id,
                    "error": "missing trusted golden proof execution",
                }
            )
            continue
        golden_failures = []
        if golden_execution.source_sha256 != record.golden_proof.sha256:
            golden_failures.append("source hash differs")
        if not golden_execution.success:
            golden_failures.append("execution failed")
        if golden_execution.repository_dependent is not True:
            golden_failures.append(
                "repository dependence was not established"
            )
        if golden_execution.deterministic is not True:
            golden_failures.append("determinism was not established")
        if not golden_execution.repository_state_unchanged:
            golden_failures.append("repository state changed")
        if golden_failures:
            errors.append(
                {
                    "question_id": record.question_id,
                    "error": (
                        "trusted golden proof is not operationally valid: "
                        + "; ".join(golden_failures)
                    ),
                }
            )
            continue
        try:
            question_metrics: dict[str, Any] = {}
            assessments.append(
                judge_answer_submission(
                    record=record,
                    golden_proof_source=bundle.golden_proofs[
                        record.golden_proof.path
                    ],
                    golden_execution=golden_execution,
                    agent_answer=answer if isinstance(answer, str) else "",
                    checkpoint_context=checkpoint_context,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_attempts=max_attempts,
                    request_timeout=request_timeout,
                    retry_delay=retry_delay,
                    metrics=question_metrics,
                )
            )
            for name, value in question_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(
                    value,
                    bool,
                ):
                    judge_metrics[name] = judge_metrics.get(name, 0) + value
        except ValueError as error:
            errors.append(
                {"question_id": record.question_id, "error": str(error)}
            )
    report = write_answer_judge_report(
        output_path,
        assessments,
        model,
        reasoning_effort,
        errors,
        cost=float(judge_metrics.get("cost", 0)),
    )
    report["metrics"] = judge_metrics
    write_json(output_path, report)
    return report


def judge_bundle_results(
    bundle_dir: Path,
    agent_results_path: Path,
    proof_executions_path: Path,
    output_path: Path,
    model: str = "openai/gpt-5.6-sol",
    reasoning_effort: str = "medium",
    max_attempts: int = 3,
    request_timeout: float | None = 90.0,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    bundle = load_generation_bundle(bundle_dir)
    results = load_agent_results(agent_results_path)
    executions = load_proof_executions(proof_executions_path)
    by_id = {}
    for result in results:
        question_id = result.get("question_id")
        question_index = result.get("question_index")
        if (
            question_id is None
            and isinstance(question_index, int)
            and 0 <= question_index < len(bundle.questions)
        ):
            question_id = bundle.questions[question_index].question_id
        if isinstance(question_id, str):
            by_id[question_id] = result
    assessments = []
    errors = []
    judge_metrics: dict[str, int | float] = {}
    for record in bundle.questions:
        result = by_id.get(record.question_id, {})
        answer = _answer_from_result(result)
        proof = _proof_submission(result)
        golden_execution = executions.get((record.question_id, "golden"))
        agent_execution = executions.get((record.question_id, "agent"))
        if golden_execution is None or agent_execution is None:
            errors.append(
                {
                    "question_id": record.question_id,
                    "error": "missing trusted proof execution",
                }
            )
            continue
        golden_failures = []
        if golden_execution.source_sha256 != record.golden_proof.sha256:
            golden_failures.append("source hash differs")
        if not golden_execution.success:
            golden_failures.append("execution failed")
        if golden_execution.repository_dependent is not True:
            golden_failures.append("repository dependence was not established")
        if golden_execution.deterministic is not True:
            golden_failures.append("determinism was not established")
        if not golden_execution.repository_state_unchanged:
            golden_failures.append("repository state changed")
        if golden_failures:
            errors.append(
                {
                    "question_id": record.question_id,
                    "error": "trusted golden proof is not operationally valid: "
                    + "; ".join(golden_failures),
                }
            )
            continue
        try:
            question_metrics: dict[str, Any] = {}
            assessments.append(
                judge_proof_submission(
                    record=record,
                    golden_proof_source=bundle.golden_proofs[record.golden_proof.path],
                    golden_execution=golden_execution,
                    agent_answer=answer if isinstance(answer, str) else "",
                    agent_proof_source=proof["script"] or "",
                    agent_execution=agent_execution,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_attempts=max_attempts,
                    request_timeout=request_timeout,
                    retry_delay=retry_delay,
                    metrics=question_metrics,
                )
            )
            for name, value in question_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    judge_metrics[name] = judge_metrics.get(name, 0) + value
        except ValueError as error:
            errors.append(
                {"question_id": record.question_id, "error": str(error)}
            )
    report = write_judge_report(
        output_path,
        assessments,
        model,
        reasoning_effort,
        errors,
        cost=float(judge_metrics.get("cost", 0)),
    )
    report["metrics"] = judge_metrics
    write_json(output_path, report)
    return report
