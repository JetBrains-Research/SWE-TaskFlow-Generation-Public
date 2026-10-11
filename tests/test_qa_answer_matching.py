import json

from interactive_benchmark_generation.qa_answer_matching import match_answer
from interactive_benchmark_generation.qa_models import (
    AnswerContract,
    FlowScope,
    QARecord,
)


def make_record(contract, accepted):
    return QARecord(
        instance_id="repo__issue",
        fact_id="fact-001",
        fact="A repository fact.",
        question_id="q-001",
        question="Return the requested canonical answer.",
        accepted_answers=(accepted,),
        incorrect_answers=("WrongOne", "WrongTwo", "WrongThree"),
        probe_path="probes/repo__issue/q-001.sh",
        answer_contract=contract,
    )


def test_symbol_reference_accepts_only_exact_owner_qualification():
    record = make_record(
        AnswerContract(
            kind="symbol_reference",
            symbol="Disabled",
            owner="Renew",
            allow_exact_qualification=True,
        ),
        "Disabled",
    )

    assert match_answer(record, "Disabled").method == "exact"
    assert match_answer(record, "Renew.Disabled").method == "answer_contract"
    assert match_answer(record, "Other.Disabled").matched is False
    assert match_answer(record, "Subscription.Renew.Disabled").matched is False
    assert match_answer(record, "Renew.DisabledValue").matched is False


def test_member_reference_requires_exact_root():
    record = make_record(
        AnswerContract(
            kind="member_reference",
            root="latestSubscription",
            member="PeriodEnd",
            allow_relative=True,
        ),
        "PeriodEnd",
    )

    assert match_answer(record, "latestSubscription.PeriodEnd").matched is True
    assert match_answer(record, "subscription.PeriodEnd").matched is False
    assert match_answer(record, "latestSubscription.Other.PeriodEnd").matched is False


def test_member_expression_compares_complete_token_structure():
    record = make_record(
        AnswerContract(
            kind="member_expression",
            root="latestSubscription",
            path="Plans?.[0]?.Title",
            allow_relative=True,
        ),
        "Plans?.[0]?.Title",
    )

    assert match_answer(
        record,
        "latestSubscription.Plans?.[0]?.Title",
    ).method == "answer_contract"
    assert match_answer(record, "subscription.Plans?.[0]?.Title").matched is False
    assert match_answer(record, "latestSubscription.Plans?.[1]?.Title").matched is False
    assert match_answer(record, "latestSubscription.Plans?.[0]?.Name").matched is False


def test_exact_contract_does_not_relax_matching():
    record = make_record(AnswerContract(kind="exact"), "Disabled")

    assert match_answer(record, "Disabled").matched is True
    assert match_answer(record, "Renew.Disabled").matched is False


def test_causal_chain_requires_complete_labeled_structure():
    record = make_record(
        AnswerContract.from_dict(
            {
                "type": "causal_chain",
                "components": [
                    {"role": "trigger", "accepted_values": ["renewal disabled"]},
                    {
                        "role": "mechanism",
                        "accepted_values": ["use current period end", "select period end"],
                    },
                    {"role": "effect", "accepted_values": ["show cancellation expiry"]},
                ],
            }
        ),
        (
            "trigger=renewal disabled; mechanism=use current period end; "
            "effect=show cancellation expiry"
        ),
    )

    assert match_answer(
        record,
        (
            "trigger=renewal disabled; mechanism=select period end; "
            "effect=show cancellation expiry"
        ),
    ).method == "answer_contract"
    assert match_answer(
        record,
        "mechanism=select period end; effect=show cancellation expiry",
    ).matched is False


def deterministic_contract(kind="causal_flow_v1"):
    if kind == "rationale_v1":
        return AnswerContract.from_dict(
            {
                "type": kind,
                "constraint": {
                    "subject": {"accepted_values": ["cached result"]},
                    "relation": {"accepted_values": ["constraint"]},
                    "object": {"accepted_values": ["None"]},
                },
                "decision": {
                    "subject": {"accepted_values": ["cache wrapper"]},
                    "relation": {"accepted_values": ["mechanism"]},
                    "object": {"accepted_values": ["controller.update"]},
                },
                "consequence": {
                    "subject": {"accepted_values": ["controller.update"]},
                    "relation": {"accepted_values": ["therefore"]},
                    "object": {"accepted_values": ["new result"]},
                },
            }
        )
    return AnswerContract.from_dict(
        {
            "type": kind,
            "trigger": {
                "subject": {"accepted_values": ["product.asin"]},
                "relation": {"accepted_values": ["triggers"]},
                "object": {
                    "accepted_values": ["B"],
                    "binding": "asin_prefix",
                },
            },
            "steps": [
                {
                    "subject": {
                        "accepted_values": ["AmazonAPI.serialize", "serialize"]
                    },
                    "relation": {"accepted_values": ["writes"]},
                    "object": {
                        "accepted_values": ["B", "C"],
                        "binding": "asin_prefix",
                    },
                }
            ],
            "outcome": {
                "subject": {
                    "accepted_values": ["Edition.from_isbn", "from_isbn"]
                },
                "relation": {"accepted_values": ["queries"]},
                "object": {"accepted_values": ["identifiers.amazon"]},
            },
        }
    )


def structured_record(contract):
    canonical = contract.canonical_answer()
    return QARecord(
        instance_id="repo__issue",
        fact_id="fact-001",
        fact="A repository fact.",
        question_id="q-001",
        question="Return one qa_answer/v1 object.",
        accepted_answers=(canonical,),
        incorrect_answers=(
            '{"protocol":"qa_answer/v1","kind":"causal_flow_v1"}',
            "{}",
            "[]",
        ),
        probe_path="probes/repo__issue/q-001.sh",
        answer_contract=contract,
        flow_scope=FlowScope("AmazonAPI.serialize", "Edition.from_isbn"),
    )


def test_causal_flow_matches_fields_without_cartesian_enumeration():
    contract = deterministic_contract()
    record = structured_record(contract)
    answer = json.loads(contract.canonical_answer())
    answer["steps"][0]["subject"] = "serialize"
    answer["outcome"]["subject"] = "from_isbn"

    match = match_answer(
        record,
        json.dumps(answer, separators=(",", ":")),
    )

    assert match.matched is True
    assert match.method == "answer_contract"


def test_causal_flow_reports_first_field_mismatch():
    contract = deterministic_contract()
    record = structured_record(contract)
    answer = json.loads(contract.canonical_answer())
    answer["outcome"]["object"] = "isbn_13"

    match = match_answer(record, json.dumps(answer))

    assert match.matched is False
    assert match.malformed is False
    assert match.reason == "field_value_not_accepted"
    assert match.mismatch_path == "$.outcome.object"


def test_causal_flow_rejects_binding_mismatch():
    contract = deterministic_contract()
    record = structured_record(contract)
    answer = json.loads(contract.canonical_answer())
    answer["steps"][0]["object"] = "C"

    match = match_answer(record, json.dumps(answer))

    assert match.reason == "binding_mismatch"
    assert match.mismatch_path == "$.steps[0].object"


def test_structured_answer_reports_malformed_json_and_shape():
    record = structured_record(deterministic_contract())

    invalid_json = match_answer(record, "{")
    extra_field = json.loads(record.answer_contract.canonical_answer())
    extra_field["extra"] = True
    invalid_shape = match_answer(record, json.dumps(extra_field))
    wrong_steps = json.loads(record.answer_contract.canonical_answer())
    wrong_steps["steps"] = []
    invalid_steps = match_answer(record, json.dumps(wrong_steps))

    assert invalid_json.malformed is True
    assert invalid_json.reason == "invalid_json"
    assert invalid_shape.malformed is True
    assert invalid_shape.reason == "invalid_answer_fields"
    assert invalid_steps.malformed is True
    assert invalid_steps.reason == "wrong_step_count"


def test_rationale_contract_matches_canonical_json():
    contract = deterministic_contract("rationale_v1")
    record = QARecord(
        instance_id="repo__issue",
        fact_id="fact-001",
        fact="A repository fact.",
        question_id="q-001",
        question="Return one rationale_v1 object.",
        accepted_answers=(contract.canonical_answer(),),
        incorrect_answers=("{}", "[]", "invalid"),
        probe_path="probes/repo__issue/q-001.sh",
        answer_contract=contract,
        flow_scope=FlowScope("cached_get", "controller.update"),
    )

    assert match_answer(record, contract.canonical_answer()).matched is True
    assert match_answer(
        record,
        (
            "trigger=renewal disabled; mechanism=select period end; "
            "effect=show future plan expiry"
        ),
    ).matched is False
