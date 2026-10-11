import time
from dataclasses import dataclass
from typing import Any

from litellm import completion

from interactive_benchmark_generation.json_utils import parse_json_object
from interactive_benchmark_generation.prompts import (
    CHECK_SYSTEM_PROMPT,
    FILTER_SYSTEM_PROMPT,
    SPLIT_SYSTEM_PROMPT,
    build_check_prompt,
    build_filter_prompt,
    build_split_prompt,
)
from interactive_benchmark_generation.validation import (
    extract_verbatim_artifacts,
    validate_parts_locally,
    validate_verbatim_placeholders,
)


@dataclass(frozen=True)
class SplitResult:
    parts: list[str]
    rationale: str
    coverage_check: str
    validation_issues: list[str]
    model: str = ""
    filter_category: str = ""
    incremental_units: tuple[str, ...] = ()
    filter_reason: str = ""


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    category: str
    units: tuple[str, ...]
    reason: str


class IneligibleProblemError(ValueError):
    def __init__(self, result: EligibilityResult):
        super().__init__(result.reason)
        self.result = result


def _message_content(response: Any) -> str:
    message = response.choices[0].message
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(message.content or "")


def call_json_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    request_timeout: float | None = None,
    reasoning_effort: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        drop_params=True,
        timeout=request_timeout,
        reasoning_effort=reasoning_effort,
    )
    if metadata is not None:
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        metadata["usage"] = usage if isinstance(usage, dict) else {}
        hidden = getattr(response, "_hidden_params", {})
        cost = hidden.get("response_cost", 0) if isinstance(hidden, dict) else 0
        metadata["cost"] = (
            cost
            if isinstance(cost, (int, float)) and not isinstance(cost, bool)
            else 0
        )
    return parse_json_object(_message_content(response))


def protect_verbatim_artifacts(problem_statement: str) -> tuple[str, dict[str, str]]:
    protected_statement = problem_statement.replace("\r\n", "\n").replace("\r", "\n")
    replacements = {}
    for index, artifact in enumerate(extract_verbatim_artifacts(protected_statement), start=1):
        if artifact.startswith("## Interface"):
            marker = "[[VERBATIM_INTERFACE_SECTION]]"
        else:
            marker = f"[[VERBATIM_ARTIFACT_{index}]]"
        protected_statement = protected_statement.replace(artifact, marker, 1)
        replacements[marker] = artifact
    return protected_statement, replacements


def restore_verbatim_artifacts(parts: list[str], replacements: dict[str, str]) -> list[str]:
    restored_parts = []
    for part in parts:
        for marker, artifact in replacements.items():
            part = part.replace(marker, artifact)
        restored_parts.append(part)
    return restored_parts


def validate_with_model(
    model: str,
    problem_statement: str,
    parts: list[str],
    incremental_units: list[str],
    request_timeout: float | None = None,
) -> list[str]:
    payload = call_json_model(
        model,
        CHECK_SYSTEM_PROMPT,
        build_check_prompt(problem_statement, parts, incremental_units),
        request_timeout,
    )
    ok = payload.get("ok")
    issues = payload.get("issues", [])
    if ok is True:
        return []
    if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
        return ["LLM validation failed without structured issues"]
    return issues


def classify_problem_statement(
    model: str,
    problem_statement: str,
    max_attempts: int,
    request_timeout: float | None = None,
    retry_delay: float = 0.0,
) -> EligibilityResult:
    previous_issues = []
    normalized_statement = problem_statement.replace("\r\n", "\n").replace("\r", "\n")
    last_error = ""
    for attempt in range(max_attempts):
        try:
            payload = call_json_model(
                model,
                FILTER_SYSTEM_PROMPT,
                build_filter_prompt(problem_statement, previous_issues),
                request_timeout,
            )
        except Exception as error:
            last_error = f"{model} eligibility call failed: {error}"
            if attempt + 1 < max_attempts and retry_delay:
                time.sleep(min(retry_delay * 2**attempt, 60.0))
            continue

        eligible = payload.get("eligible")
        category = payload.get("category")
        reason = payload.get("reason")
        raw_units = payload.get("units", [])
        issues = []
        if not isinstance(eligible, bool):
            issues.append("eligible must be a boolean")
        if category not in {"atomic_evidence_only", "incremental_requirements", "composite_subtasks"}:
            issues.append("category must be one of the requested values")
        if not isinstance(reason, str) or not reason.strip():
            issues.append("reason must be a non-empty string")

        units = []
        if eligible is True:
            if category == "atomic_evidence_only":
                issues.append("eligible atomic_evidence_only classifications are inconsistent")
            if not isinstance(raw_units, list) or len(raw_units) < 2:
                issues.append("eligible issues must contain at least two units")
            else:
                seen_requirements = set()
                for index, raw_unit in enumerate(raw_units, start=1):
                    if not isinstance(raw_unit, dict):
                        issues.append(f"unit {index} must be an object")
                        continue
                    requirement = raw_unit.get("requirement")
                    evidence = raw_unit.get("source_evidence")
                    if not isinstance(requirement, str) or not requirement.strip():
                        issues.append(f"unit {index} requirement must be a non-empty string")
                        continue
                    normalized_requirement = " ".join(requirement.lower().split())
                    if normalized_requirement in seen_requirements:
                        issues.append(f"unit {index} duplicates an earlier requirement")
                    seen_requirements.add(normalized_requirement)
                    if not isinstance(evidence, str) or not evidence.strip():
                        issues.append(f"unit {index} source_evidence must be a non-empty exact quote")
                    elif evidence.replace("\r\n", "\n").replace("\r", "\n") not in normalized_statement:
                        issues.append(f"unit {index} source_evidence is not an exact quote from problem_statement")
                    units.append(requirement.strip())
        elif eligible is False and category != "atomic_evidence_only":
            issues.append("ineligible issues must use category atomic_evidence_only")

        if not issues:
            return EligibilityResult(
                eligible=eligible,
                category=category,
                units=tuple(units),
                reason=reason.strip(),
            )
        previous_issues = issues
        last_error = "; ".join(issues)

    raise ValueError(f"Could not classify problem_statement eligibility: {last_error}")


def split_problem_statement(
    problem_statement: str,
    n_parts: int,
    model: str,
    max_attempts: int,
    llm_validation: bool,
    fallback_models: list[str] | None = None,
    request_timeout: float | None = None,
    retry_delay: float = 0.0,
) -> SplitResult:
    last_issues = []
    eligibility = classify_problem_statement(
        model,
        problem_statement,
        max_attempts,
        request_timeout,
        retry_delay,
    )
    if not eligibility.eligible:
        raise IneligibleProblemError(eligibility)
    protected_statement, verbatim_replacements = protect_verbatim_artifacts(problem_statement)
    models = [model, *(fallback_models or [])]
    for model_name in dict.fromkeys(models):
        model_issues = []
        previous_parts = []
        for attempt in range(max_attempts):
            try:
                payload = call_json_model(
                    model_name,
                    SPLIT_SYSTEM_PROMPT,
                    build_split_prompt(
                        protected_statement,
                        n_parts,
                        model_issues,
                        list(verbatim_replacements),
                        list(eligibility.units),
                        previous_parts,
                    ),
                    request_timeout,
                )
            except Exception as error:
                last_issues = [f"{model_name} call failed: {error}"]
                if attempt + 1 < max_attempts and retry_delay:
                    time.sleep(min(retry_delay * 2**attempt, 60.0))
                continue

            raw_parts = payload.get("parts")
            if not isinstance(raw_parts, list):
                last_issues = [f"{model_name} response did not contain a parts list"]
                continue

            normalized_raw_parts = [str(part).strip() for part in raw_parts]
            placeholder_result = validate_verbatim_placeholders(
                normalized_raw_parts,
                list(verbatim_replacements),
            )
            parts = restore_verbatim_artifacts(
                normalized_raw_parts,
                verbatim_replacements,
            )
            local_result = validate_parts_locally(problem_statement, parts, n_parts)
            issues = [*placeholder_result.issues, *local_result.issues]
            if not issues and llm_validation:
                try:
                    issues.extend(
                        validate_with_model(
                            model_name,
                            problem_statement,
                            parts,
                            list(eligibility.units),
                            request_timeout,
                        )
                    )
                except Exception as error:
                    issues.append(f"{model_name} validation call failed: {error}")
                    if attempt + 1 < max_attempts and retry_delay:
                        time.sleep(min(retry_delay * 2**attempt, 60.0))

            if not issues:
                return SplitResult(
                    parts=parts,
                    rationale=str(payload.get("rationale", "")),
                    coverage_check=str(payload.get("coverage_check", "")),
                    validation_issues=[],
                    model=model_name,
                    filter_category=eligibility.category,
                    incremental_units=eligibility.units,
                    filter_reason=eligibility.reason,
                )

            model_issues = issues
            previous_parts = normalized_raw_parts
            last_issues = [f"{model_name}: {issue}" for issue in issues]

    raise ValueError(f"Could not generate a valid split with configured models: {'; '.join(last_issues)}")
