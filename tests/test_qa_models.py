import pytest

from interactive_benchmark_generation.qa_models import (
    AnswerContract,
    Fact,
    FlowScope,
    GenerationBundle,
    ProbeValidation,
    QARecord,
    normalize_answer,
)


def make_record(**overrides):
    values = {
        "instance_id": "repo__issue",
        "fact_id": "fact-001",
        "fact": "The command emits a scalar.",
        "question_id": "q-001",
        "question": "What does the command emit?",
        "accepted_answers": ("Café value",),
        "incorrect_answers": ("Error", "None", "Other"),
        "probe_path": "probes/repo__issue/q-001.sh",
    }
    values.update(overrides)
    return QARecord(**values)


def test_normalize_answer_uses_nfkc_whitespace_and_casefold():
    assert normalize_answer("  ＣＡＦÉ\tValue  ") == normalize_answer("café value")


def test_numeric_major_schema_version_is_normalized():
    fact = Fact("repo__issue", "fact-001", "A fact", schema_version=1)

    assert fact.schema_version == "1.0"
    assert fact.to_dict()["schema_version"] == "1.0"


def test_named_bundle_schema_version_is_normalized():
    fact = Fact(
        "repo__issue",
        "fact-001",
        "A fact",
        schema_version="qa_benchmark_bundle/v1",
    )

    assert fact.schema_version == "1.0"


def test_structured_documentation_is_serialized():
    bundle = GenerationBundle.from_dict(
        {
            "instance_id": "repo__issue",
            "documentation": {"summary": "A summary", "evidence": ["A source"]},
            "facts": [{"fact_id": "fact-001", "fact": "The command emits a scalar."}],
            "questions": [make_record().to_dict()],
            "probes": {"probes/repo__issue/q-001.sh": "echo PASSED"},
        }
    )

    assert '"summary": "A summary"' in bundle.documentation
    assert '"evidence": [' in bundle.documentation


def test_qa_record_requires_three_distinct_incorrect_answers():
    with pytest.raises(ValueError, match="exactly three"):
        make_record(incorrect_answers=("one", "two"))

    with pytest.raises(ValueError, match="must not overlap"):
        make_record(incorrect_answers=(" café  value ", "two", "three"))


def test_answer_contract_round_trip():
    contract = AnswerContract(
        kind="symbol_reference",
        symbol="Disabled",
        owner="Renew",
        allow_exact_qualification=True,
    )

    assert AnswerContract.from_dict(contract.to_dict()) == contract


def test_answer_contract_rejects_fields_from_another_kind():
    with pytest.raises(ValueError, match="does not accept fields"):
        AnswerContract(kind="symbol_reference", symbol="Disabled", owner="Renew", root="other")


@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        (
            "causal_flow_v1",
            {
                "trigger": {
                    "subject": "request",
                    "relation": "triggers",
                    "object": "parser",
                },
                "steps": [
                    {
                        "subject": "parser",
                        "relation": "calls",
                        "object": "model",
                    }
                ],
                "outcome": {
                    "subject": "model",
                    "relation": "returns",
                    "object": "record",
                },
            },
        ),
        (
            "rationale_v1",
            {
                "constraint": {
                    "subject": "cache result",
                    "relation": "constraint",
                    "object": "None",
                },
                "decision": {
                    "subject": "cache wrapper",
                    "relation": "mechanism",
                    "object": "refresh",
                },
                "consequence": {
                    "subject": "refresh",
                    "relation": "therefore",
                    "object": "new result",
                },
            },
        ),
    ],
)
def test_deterministic_contract_round_trip(kind, fields):
    contract = AnswerContract.from_dict({"type": kind, **fields})

    assert AnswerContract.from_dict(contract.to_dict()) == contract
    assert contract.canonical_answer().startswith(
        f'{{"protocol":"qa_answer/v1","kind":"{kind}"'
    )


def test_deterministic_contract_rejects_unknown_relation():
    with pytest.raises(ValueError, match="Unsupported causal_flow_v1 relation"):
        AnswerContract.from_dict(
            {
                "type": "causal_flow_v1",
                "trigger": {
                    "subject": "request",
                    "relation": "explains",
                    "object": "parser",
                },
                "steps": [
                    {
                        "subject": "parser",
                        "relation": "calls",
                        "object": "model",
                    }
                ],
                "outcome": {
                    "subject": "model",
                    "relation": "returns",
                    "object": "record",
                },
            }
        )


def test_deterministic_contract_accepts_schema_wrapper():
    contract = AnswerContract.from_dict(
        {
            "type": "causal_flow_v1",
            "schema": {
                "protocol": {"accepted_values": ["qa_answer/v1"]},
                "kind": {"accepted_values": ["causal_flow_v1"]},
                "trigger": {
                    "subject": "request",
                    "relation": "triggers",
                    "object": "parser",
                },
                "steps": [
                    {
                        "subject": "parser",
                        "relation": "calls",
                        "object": "model",
                    }
                ],
                "outcome": {
                    "subject": "model",
                    "relation": "returns",
                    "object": "record",
                },
            },
        }
    )

    assert contract.kind.value == "causal_flow_v1"
    assert contract.canonical_answer().startswith('{"protocol":"qa_answer/v1"')


def test_deterministic_question_requires_flow_scope():
    contract = AnswerContract.from_dict(
        {
            "type": "rationale_v1",
            "constraint": {
                "subject": "cache result",
                "relation": "constraint",
                "object": "None",
            },
            "decision": {
                "subject": "cache wrapper",
                "relation": "mechanism",
                "object": "refresh",
            },
            "consequence": {
                "subject": "refresh",
                "relation": "therefore",
                "object": "new result",
            },
        }
    )

    with pytest.raises(ValueError, match="require flow_scope"):
        make_record(answer_contract=contract)

    record = make_record(
        answer_contract=contract,
        flow_scope=FlowScope("cached_get", "controller.update"),
    )
    assert record.flow_scope.entry_symbol == "cached_get"


def test_generation_bundle_requires_one_unique_fact_per_question():
    fact = Fact("repo__issue", "fact-001", "The command emits a scalar.")
    record = make_record()
    second = make_record(question_id="q-002", probe_path="probes/repo__issue/q-002.sh")

    with pytest.raises(ValueError, match="at most one question"):
        GenerationBundle(
            instance_id="repo__issue",
            documentation="Documentation",
            facts=(fact,),
            questions=(record, second),
            probes={
                record.probe_path: "echo PASSED",
                second.probe_path: "echo PASSED",
            },
        )


def test_probe_validation_round_trip():
    validation = ProbeValidation(
        accepted_passed=True,
        incorrect_rejected=True,
        runs=({"answer": "yes", "passed": True},),
    )

    assert ProbeValidation.from_dict(validation.to_dict()) == validation
