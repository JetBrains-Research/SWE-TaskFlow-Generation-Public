import pytest

from interactive_benchmark_generation.llm import (
    IneligibleProblemError,
    protect_verbatim_artifacts,
    restore_verbatim_artifacts,
    split_problem_statement,
)

PROBLEM_STATEMENT = (
    "Nested model composition must work during evaluation in arithmetic expressions. "
    "Evaluation must also preserve the input model order."
)


def eligibility_payload():
    return {
        "eligible": True,
        "category": "incremental_requirements",
        "units": [
            {
                "requirement": "Support nested model composition in arithmetic expressions.",
                "source_evidence": ("Nested model composition must work during evaluation in arithmetic expressions."),
            },
            {
                "requirement": "Preserve input model order during evaluation.",
                "source_evidence": "Evaluation must also preserve the input model order.",
            },
        ],
        "reason": "The issue contains two distinct acceptance conditions.",
    }


def test_verbatim_artifacts_are_protected_and_restored():
    artifact = "```shell\nfirst output line\nsecond output line\n```"
    problem_statement = f"Run the command.\r\n\r\n{artifact.replace(chr(10), chr(13) + chr(10))}\r\n\r\nIt should pass."

    protected, replacements = protect_verbatim_artifacts(problem_statement)
    restored = restore_verbatim_artifacts(
        ["Run the command.", "Here is the output:\n[[VERBATIM_ARTIFACT_1]]", "It should pass."],
        replacements,
    )

    assert artifact not in protected
    assert "[[VERBATIM_ARTIFACT_1]]" in protected
    assert artifact in restored[1]


def test_interface_section_is_protected_as_one_named_artifact():
    interface = "## Interface\n\nAdd `Exporter.Export(ctx, writer) error`."
    protected, replacements = protect_verbatim_artifacts(
        f"Implement YAML export.\n\n## Requirements\n\nExport documents.\n\n{interface}"
    )
    restored = restore_verbatim_artifacts(
        ["Implement YAML export.", "Export documents.", "[[VERBATIM_INTERFACE_SECTION]]"],
        replacements,
    )

    assert protected.endswith("[[VERBATIM_INTERFACE_SECTION]]")
    assert replacements == {"[[VERBATIM_INTERFACE_SECTION]]": interface}
    assert restored[2] == interface


def test_split_uses_generation_model_for_validation(monkeypatch):
    calls = []

    def fake_call_json_model(model, system_prompt, _user_prompt, _request_timeout):
        calls.append(model)
        if "classify" in system_prompt:
            return eligibility_payload()
        if "transform" in system_prompt:
            return {
                "parts": [
                    "Nested model composition must work correctly during evaluation.",
                    "The affected context is arithmetic expressions with nested models.",
                    "Evaluation must also preserve the input model order.",
                ],
                "rationale": "test",
                "coverage_check": "test",
            }
        return {"ok": True, "issues": []}

    monkeypatch.setattr("interactive_benchmark_generation.llm.call_json_model", fake_call_json_model)
    result = split_problem_statement(
        problem_statement=PROBLEM_STATEMENT,
        n_parts=3,
        model="openai/gpt-5.4-mini",
        max_attempts=1,
        llm_validation=True,
    )

    assert calls == ["openai/gpt-5.4-mini", "openai/gpt-5.4-mini", "openai/gpt-5.4-mini"]
    assert result.model == "openai/gpt-5.4-mini"
    assert result.filter_category == "incremental_requirements"
    assert len(result.incremental_units) == 2


def test_split_retry_receives_previous_parts_with_unexpanded_placeholders(monkeypatch):
    artifact = "```text\noriginal artifact contents\n```"
    problem_statement = f"{PROBLEM_STATEMENT}\n\n{artifact}"
    generation_prompts = []

    def fake_call_json_model(_model, system_prompt, user_prompt, _request_timeout):
        if "classify" in system_prompt:
            return eligibility_payload()
        if "transform" in system_prompt:
            generation_prompts.append(user_prompt)
            if len(generation_prompts) == 1:
                return {
                    "parts": [
                        "Nested model composition must work correctly during arithmetic expression evaluation.",
                        "Use this exact diagnostic:\n[[VERBATIM_ARTIFACT_1]]",
                        "Preserve the input model order.\n[[VERBATIM_ARTIFACT_1]]",
                    ],
                    "rationale": "test",
                    "coverage_check": "test",
                }
            return {
                "parts": [
                    "Nested model composition must work correctly during arithmetic expression evaluation.",
                    "Use this exact diagnostic:\n[[VERBATIM_ARTIFACT_1]]",
                    "Evaluation must also preserve the input model order.",
                ],
                "rationale": "test",
                "coverage_check": "test",
            }
        return {"ok": True, "issues": []}

    monkeypatch.setattr("interactive_benchmark_generation.llm.call_json_model", fake_call_json_model)
    split_problem_statement(
        problem_statement=problem_statement,
        n_parts=3,
        model="openai/gpt-5.4-mini",
        max_attempts=2,
        llm_validation=False,
    )

    assert len(generation_prompts) == 2
    assert artifact not in generation_prompts[1]
    assert generation_prompts[1].count("[[VERBATIM_ARTIFACT_1]]") >= 3
    assert "must appear exactly once, found 2" in generation_prompts[1]


def test_split_supports_five_parts(monkeypatch):
    requirements = [
        "Support nested model composition throughout the model evaluation pipeline.",
        "Evaluate nested models in arithmetic expressions.",
        "Preserve the input model order.",
        "Handle deeply nested model structures.",
        "Keep evaluation results deterministic.",
    ]
    problem_statement = " ".join(requirements)

    def fake_call_json_model(_model, system_prompt, _user_prompt, _request_timeout):
        if "classify" in system_prompt:
            return {
                "eligible": True,
                "category": "incremental_requirements",
                "units": [
                    {
                        "requirement": requirement,
                        "source_evidence": requirement,
                    }
                    for requirement in requirements
                ],
                "reason": "The issue contains five distinct acceptance conditions.",
            }
        if "transform" in system_prompt:
            return {
                "parts": requirements,
                "rationale": "Each part introduces one requirement.",
                "coverage_check": "All five requirements are present.",
            }
        return {"ok": True, "issues": []}

    monkeypatch.setattr("interactive_benchmark_generation.llm.call_json_model", fake_call_json_model)
    result = split_problem_statement(
        problem_statement=problem_statement,
        n_parts=5,
        model="openai/gpt-5.4-mini",
        max_attempts=1,
        llm_validation=True,
    )

    assert result.parts == requirements
    assert len(result.parts) == 5


def test_split_retries_transient_model_failure(monkeypatch):
    call_count = 0
    delays = []

    def fake_call_json_model(_model, system_prompt, _user_prompt, _request_timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("no healthy deployments")
        if "classify" in system_prompt:
            return eligibility_payload()
        if "transform" in system_prompt:
            return {
                "parts": [
                    "Nested model composition must work correctly during evaluation.",
                    "The affected context is arithmetic expressions with nested models.",
                    "Evaluation must also preserve the input model order.",
                ],
                "rationale": "test",
                "coverage_check": "test",
            }
        return {"ok": True, "issues": []}

    monkeypatch.setattr("interactive_benchmark_generation.llm.call_json_model", fake_call_json_model)
    monkeypatch.setattr("interactive_benchmark_generation.llm.time.sleep", delays.append)
    result = split_problem_statement(
        problem_statement=PROBLEM_STATEMENT,
        n_parts=3,
        model="openai/gpt-5.6-luna",
        max_attempts=2,
        llm_validation=True,
        retry_delay=5.0,
    )

    assert call_count == 4
    assert delays == [5.0]
    assert result.model == "openai/gpt-5.6-luna"


def test_split_filters_atomic_evidence_only_issue(monkeypatch):
    def fake_call_json_model(_model, _system_prompt, _user_prompt, _request_timeout):
        return {
            "eligible": False,
            "category": "atomic_evidence_only",
            "units": [],
            "reason": "The examples only demonstrate one requested behavioral change.",
        }

    monkeypatch.setattr("interactive_benchmark_generation.llm.call_json_model", fake_call_json_model)

    with pytest.raises(IneligibleProblemError, match="one requested behavioral change"):
        split_problem_statement(
            problem_statement="srepr should print dict and set elements correctly.",
            n_parts=3,
            model="openai/gpt-5.6-luna",
            max_attempts=1,
            llm_validation=True,
        )
