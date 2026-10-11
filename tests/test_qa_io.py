import json

import pytest

from interactive_benchmark_generation.qa_io import (
    load_generation_bundle,
    load_qa_records,
    validate_answer_contracts,
    validate_question_complexity,
    write_qa_bundle,
)
from interactive_benchmark_generation.qa_models import GenerationBundle
from interactive_benchmark_generation.trajectory import load_trajectory, summarize_trajectory


def bundle_value():
    return {
        "schema_version": "1.0",
        "instance_id": "repo__issue",
        "documentation": "# Repository\n\nA documented fact.",
        "facts": [
            {
                "fact_id": "fact-001",
                "fact": "The command prints four.",
            }
        ],
        "questions": [
            {
                "fact_id": "fact-001",
                "question_id": "q-001",
                "question": "What does the command print?",
                "accepted_answers": ["4"],
                "incorrect_answers": ["1", "2", "Error"],
                "answer_contract": {"type": "exact"},
                "probe": "#!/usr/bin/env bash\n[[ \"$1\" == \"4\" ]] && echo PASSED || echo FAILED",
            }
        ],
        "metadata": {"generator": "structured"},
        "debug": {"attempts": 1},
    }


def add_complexity(value):
    question = value["questions"][0]
    question["category"] = "why_design_rationale"
    question["evidence_hops"] = [
        {"evidence_id": "e1", "file": "a.py", "symbol": "entry", "claim": "Entry selects."},
        {"evidence_id": "e2", "file": "b.py", "symbol": "middle", "claim": "Middle transforms."},
        {"evidence_id": "e3", "file": "b.py", "symbol": "exit", "claim": "Exit returns."},
    ]
    question["causal_chain"] = [
        {"step_id": "s1", "claim": "Selection occurs.", "evidence_ids": ["e1"]},
        {"step_id": "s2", "claim": "Transformation occurs.", "evidence_ids": ["e2"]},
        {"step_id": "s3", "claim": "Result is returned.", "evidence_ids": ["e3"]},
    ]
    return value


def test_write_qa_bundle_creates_contract_artifacts_and_loads_trajectory(tmp_path):
    generation_path = tmp_path / "generation.json"
    generation_path.write_text(json.dumps(bundle_value()))
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "info": {
                    "config": {"model": {"model_name": "openai/test"}},
                    "exit_status": "Submitted",
                    "submission": "patch",
                },
                "messages": [{"role": "user", "content": "task"}],
            }
        )
    )
    output_dir = tmp_path / "output"

    bundle = load_generation_bundle(generation_path)
    write_qa_bundle(bundle, output_dir, trajectory_path)

    assert {path.name for path in output_dir.iterdir()} == {
        "debug.json",
        "documentation.md",
        "facts.json",
        "metadata.json",
        "probes",
        "questions.jsonl",
    }
    records = load_qa_records(output_dir)
    assert records[0].accepted_answers == ("4",)
    assert (output_dir / records[0].probe_path).stat().st_mode & 0o111
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["trajectory"]["model"] == "openai/test"
    assert metadata["trajectory"]["message_count"] == 1


def test_load_known_trajectory_shape_generically(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "info": {"model_stats": {"api_calls": 2}},
                "messages": [{"role": "system"}, {"role": "user"}],
                "extra": {"future": True},
            }
        )
    )

    trajectory = load_trajectory(path)
    summary = summarize_trajectory(trajectory)

    assert trajectory.raw["extra"] == {"future": True}
    assert summary["message_count"] == 2
    assert summary["model_stats"] == {"api_calls": 2}


def test_generation_bundle_accepts_explicit_probe_mapping():
    value = bundle_value()
    script = value["questions"][0].pop("probe")
    value["probes"] = {"probes/repo__issue/q-001.sh": script}

    bundle = GenerationBundle.from_dict(value)

    assert bundle.probes["probes/repo__issue/q-001.sh"] == script


def test_validate_answer_contracts_requires_every_contract():
    value = bundle_value()
    value["questions"][0].pop("answer_contract")
    bundle = GenerationBundle.from_dict(value)

    with pytest.raises(ValueError, match="Missing answer_contract"):
        validate_answer_contracts(bundle)


def test_validate_answer_contracts_rejects_structurally_accepted_incorrect_answer():
    value = bundle_value()
    question = value["questions"][0]
    question["accepted_answers"] = ["Disabled"]
    question["incorrect_answers"] = ["Renew.Disabled", "Enabled", "Unknown"]
    question["answer_contract"] = {
        "type": "symbol_reference",
        "symbol": "Disabled",
        "owner": "Renew",
        "allow_exact_qualification": True,
    }
    bundle = GenerationBundle.from_dict(value)

    with pytest.raises(ValueError, match="accepts incorrect answers"):
        validate_answer_contracts(bundle)


def test_validate_answer_contracts_accepts_structured_canonical_answer():
    value = bundle_value()
    question = value["questions"][0]
    question["flow_scope"] = {
        "entry_symbol": "request",
        "terminal_symbol": "model",
    }
    question["answer_contract"] = {
        "type": "causal_flow_v1",
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
    }
    contract = GenerationBundle.from_dict(value).questions[0].answer_contract
    question["accepted_answers"] = [contract.canonical_answer()]
    incorrect = []
    for field, replacement in (
        ("subject", "other request"),
        ("relation", "selects"),
        ("object", "other parser"),
    ):
        answer = json.loads(contract.canonical_answer())
        answer["trigger"][field] = replacement
        incorrect.append(json.dumps(answer, separators=(",", ":")))
    question["incorrect_answers"] = incorrect
    bundle = GenerationBundle.from_dict(value)

    validate_answer_contracts(bundle)

    value["questions"][0]["incorrect_answers"][0] = "{}"
    malformed_bundle = GenerationBundle.from_dict(value)
    with pytest.raises(ValueError, match="valid protocol JSON"):
        validate_answer_contracts(malformed_bundle)


def test_validate_question_complexity_accepts_multihop_question():
    value = add_complexity(bundle_value())
    value["questions"][0]["question"] = "Why does the entry path transform the value before returning it?"
    bundle = GenerationBundle.from_dict(value)

    validate_question_complexity(bundle)


def test_validate_question_complexity_rejects_lookup_question():
    value = add_complexity(bundle_value())
    value["questions"][0]["question"] = "Which property stores the value?"
    bundle = GenerationBundle.from_dict(value)

    with pytest.raises(ValueError, match="Lookup question"):
        validate_question_complexity(bundle)
