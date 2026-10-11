import json
import subprocess
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from interactive_benchmark_generation.cli import app
from interactive_benchmark_generation.qa_evaluation import (
    evaluate_agent_results,
    evaluate_agent_run,
)
from interactive_benchmark_generation.qa_io import load_qa_records, write_qa_bundle
from interactive_benchmark_generation.qa_models import (
    AdversarialReview,
    GenerationBundle,
    ProbeValidation,
)
from interactive_benchmark_generation.qa_validation import (
    ADVERSARIAL_REVIEW_CHECKS,
    ADVERSARIAL_REVIEW_CHECKS_V1,
    filter_validated_bundle,
    import_adversarial_review,
    import_probe_validation_report,
    validate_qa_bundle,
)


def test_legacy_adversarial_review_check_export_is_unchanged():
    assert ADVERSARIAL_REVIEW_CHECKS is ADVERSARIAL_REVIEW_CHECKS_V1
    assert {
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
    } == ADVERSARIAL_REVIEW_CHECKS_V1


def make_bundle(script, answer_contract=None):
    question = {
        "fact_id": "fact-001",
        "question_id": "q-001",
        "question": "What value is stored?",
        "accepted_answers": ["4", " \uff14 "],
        "incorrect_answers": ["1", "2", "Error"],
        "probe": script,
    }
    if answer_contract is not None:
        question["answer_contract"] = answer_contract
    return GenerationBundle.from_dict(
        {
            "instance_id": "repo__issue",
            "documentation": "The value file contains four.",
            "facts": [
                {
                    "fact_id": "fact-001",
                    "fact": "The value file contains four.",
                }
            ],
            "questions": [question],
        }
    )


def make_repository(path):
    path.mkdir()
    (path / "value.txt").write_text("4\n")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "value.txt"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=path,
        check=True,
    )


def test_validate_bundle_runs_every_answer_and_records_evidence(tmp_path):
    repository = tmp_path / "repository"
    make_repository(repository)
    bundle_dir = tmp_path / "bundle"
    script = (
        "#!/usr/bin/env bash\n"
        "result=$(cat value.txt)\n"
        "if [[ \"$1\" == \"$result\" || \"$1\" == \" \uff14 \" ]]; then\n"
        "    echo PASSED\n"
        "else\n"
        "    echo FAILED\n"
        "fi"
    )
    write_qa_bundle(make_bundle(script), bundle_dir)

    report = validate_qa_bundle(bundle_dir, repository, timeout=2)

    assert report["valid"] is True
    assert report["repository_state_unchanged"] is True
    assert len(report["questions"][0]["runs"]) == 5
    record = load_qa_records(bundle_dir)[0]
    assert record.probe_validation is not None
    assert record.probe_validation.accepted_passed is True
    assert (bundle_dir / "probe-validation.json").exists()


def test_validate_bundle_rejects_non_exact_probe_output(tmp_path):
    repository = tmp_path / "repository"
    make_repository(repository)
    bundle_dir = tmp_path / "bundle"
    script = "#!/usr/bin/env bash\necho \"PASSED extra\""
    write_qa_bundle(make_bundle(script), bundle_dir)

    report = validate_qa_bundle(bundle_dir, repository, timeout=2)

    assert report["valid"] is False
    assert report["questions"][0]["accepted_passed"] is False


def test_evaluate_compares_normalized_answers_and_writes_reports(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle("#!/usr/bin/env bash\necho PASSED"), bundle_dir)
    agent_results = tmp_path / "answers.jsonl"
    agent_results.write_text(
        json.dumps(
            {
                "question_id": "q-001",
                "answer": "  \uff14\t",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cached_tokens": 3,
                },
                "cost": 0.02,
                "steps": 2,
                "tool_calls": 1,
                "latency_seconds": 0.5,
            }
        )
        + "\n"
    )
    output_dir = tmp_path / "report"

    report = evaluate_agent_results(bundle_dir, agent_results, output_dir)

    assert report["n_correct"] == 1
    assert report["questions"][0]["normalized_answer"] == "4"
    assert report["totals"]["input_tokens"] == 10
    assert report["average_steps_per_answer"] == 2
    assert (output_dir / "evaluation.json").exists()
    assert "Steps: 2" in (output_dir / "evaluation.md").read_text()
    assert "Result: CORRECT" in (output_dir / "evaluation.md").read_text()


def test_evaluate_uses_structural_answer_contract(tmp_path):
    bundle = GenerationBundle.from_dict(
        {
            "instance_id": "repo__issue",
            "documentation": "Subscription facts.",
            "facts": [{"fact_id": "fact-001", "fact": "Renew.Disabled disables renewal."}],
            "questions": [
                {
                    "fact_id": "fact-001",
                    "question_id": "q-001",
                    "question": "Return only the Renew enum member name.",
                    "accepted_answers": ["Disabled"],
                    "incorrect_answers": ["Enabled", "None", "Unknown"],
                    "answer_contract": {
                        "type": "symbol_reference",
                        "symbol": "Disabled",
                        "owner": "Renew",
                        "allow_exact_qualification": True,
                    },
                    "probe": "#!/usr/bin/env bash\necho PASSED",
                }
            ],
        }
    )
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    results = tmp_path / "answers.jsonl"
    results.write_text(json.dumps({"question_id": "q-001", "answer": "Renew.Disabled"}) + "\n")

    report = evaluate_agent_results(bundle_dir, results, tmp_path / "report")

    assert report["n_correct"] == 1
    assert report["questions"][0]["match_method"] == "answer_contract"


def test_evaluate_reports_structured_mismatch_path_and_malformed_reason(tmp_path):
    value = {
        "instance_id": "repo__issue",
        "documentation": "A request reaches a model.",
        "facts": [{"fact_id": "fact-001", "fact": "A request reaches a model."}],
        "questions": [
            {
                "fact_id": "fact-001",
                "question_id": "q-001",
                "question": "Return one causal_flow_v1 object ending at model.",
                "flow_scope": {
                    "entry_symbol": "request",
                    "terminal_symbol": "model",
                },
                "answer_contract": {
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
                },
                "accepted_answers": ["placeholder"],
                "incorrect_answers": ["{}", "[]", "invalid"],
                "probe": "#!/usr/bin/env bash\necho PASSED",
            }
        ],
    }
    initial = GenerationBundle.from_dict(value)
    canonical = initial.questions[0].answer_contract.canonical_answer()
    value["questions"][0]["accepted_answers"] = [canonical]
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(GenerationBundle.from_dict(value), bundle_dir)
    results = tmp_path / "answers.jsonl"
    results.write_text(json.dumps({"question_id": "q-001", "answer": "{"}) + "\n")

    report = evaluate_agent_results(bundle_dir, results, tmp_path / "report")

    question = report["questions"][0]
    assert question["correct"] is False
    assert question["malformed"] is True
    assert question["match_reason"] == "invalid_json"
    assert "Match reason: invalid_json" in (
        tmp_path / "report/evaluation.md"
    ).read_text()


def test_evaluate_reads_sequential_answer_trajectory(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle("#!/usr/bin/env bash\necho PASSED"), bundle_dir)
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "info": {
                    "answer_records": [
                        {
                            "question_index": 0,
                            "question": "What is the value?",
                            "answer": "4",
                        }
                    ]
                },
                "messages": [
                    {"role": "user", "content": "What is the value?"},
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [{"command": "cat value.txt"}],
                            "cost": 0.03,
                            "response": {
                                "usage": {
                                    "prompt_tokens": 12,
                                    "completion_tokens": 2,
                                    "prompt_tokens_details": {"cached_tokens": 5},
                                }
                            },
                        },
                    },
                ],
            }
        )
        + "\n"
    )

    report = evaluate_agent_results(bundle_dir, trajectory, tmp_path / "report")

    assert report["n_correct"] == 1
    assert report["unknown_results"] == []
    assert report["totals"]["input_tokens"] == 12
    assert report["totals"]["output_tokens"] == 2
    assert report["totals"]["cached_tokens"] == 5
    assert report["totals"]["cost"] == 0.03
    assert report["totals"]["steps"] == 1
    assert report["average_steps_per_answer"] == 1
    assert report["totals"]["tool_calls"] == 1


def test_evaluate_agent_run_aggregates_instance_reports(tmp_path):
    bundle_root = tmp_path / "bundles"
    write_qa_bundle(make_bundle("#!/usr/bin/env bash\necho PASSED"), bundle_root / "instances/repo__issue")
    trial_dir = tmp_path / "agent-run/trial"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps({"task_name": "cais/repo__issue"}) + "\n")
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "info": {
                    "answer_records": [
                        {
                            "question_index": 0,
                            "question": "What is the value?",
                            "answer": "4",
                        }
                    ]
                }
            }
        )
        + "\n"
    )

    report = evaluate_agent_run(bundle_root, tmp_path / "agent-run", tmp_path / "report")

    assert report["n_instances"] == 1
    assert report["n_correct"] == 1
    assert report["average_steps_per_answer"] == 0
    assert (tmp_path / "report/evaluation.md").is_file()


def test_import_probe_validation_report_updates_records(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle("#!/usr/bin/env bash\necho PASSED"), bundle_dir)
    report_path = tmp_path / "output.json"
    report_path.write_text(
        json.dumps(
            {
                "valid": True,
                "results": [
                    {
                        "question_id": "q-001",
                        "accepted": [
                            {"answer": "4", "passed": True},
                            {"answer": " \uff14 ", "passed": True},
                        ],
                        "incorrect": [
                            {"answer": "1", "rejected": True},
                            {"answer": "2", "rejected": True},
                            {"answer": "Error", "rejected": True},
                        ],
                    }
                ],
            }
        )
        + "\n"
    )

    report = import_probe_validation_report(bundle_dir, report_path)
    record = load_qa_records(bundle_dir)[0]

    assert report["valid"] is True
    assert record.probe_validation is not None
    assert record.probe_validation.accepted_passed is True
    assert record.probe_validation.incorrect_rejected is True


def test_import_adversarial_review_updates_records(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle("#!/usr/bin/env bash\necho PASSED"), bundle_dir)
    report_path = tmp_path / "review.json"
    report_path.write_text(
        json.dumps(
            {
                "instance_id": "repo__issue",
                "reviews": [
                    {
                        "question_id": "q-001",
                        "accepted": True,
                        "checks": dict.fromkeys(ADVERSARIAL_REVIEW_CHECKS, True),
                        "reasons": [],
                    }
                ]
            }
        )
        + "\n"
    )

    imported = import_adversarial_review(bundle_dir, report_path)

    assert imported["n_accepted"] == 1
    record = load_qa_records(bundle_dir)[0]
    assert record.adversarial_review is not None
    assert record.adversarial_review.accepted is True


def test_import_rejected_review_requires_structured_diagnostics(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(make_bundle("#!/usr/bin/env bash\necho PASSED"), bundle_dir)
    checks = dict.fromkeys(ADVERSARIAL_REVIEW_CHECKS, True)
    checks["alternative_grounded_answer_absent"] = False
    report = {
        "instance_id": "repo__issue",
        "reviews": [
            {
                "question_id": "q-001",
                "accepted": False,
                "checks": checks,
                "reasons": ["A second endpoint also satisfies the wording."],
            }
        ],
    }
    report_path = tmp_path / "review.json"
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="lacks structured diagnostics"):
        import_adversarial_review(bundle_dir, report_path)

    report["reviews"][0]["rejection_codes"] = ["alternative_grounded_path"]
    report["reviews"][0]["defeating_paths"] = ["other.py:OtherModel.lookup"]
    report_path.write_text(json.dumps(report))
    imported = import_adversarial_review(bundle_dir, report_path)

    assert imported["n_accepted"] == 0
    review = load_qa_records(bundle_dir)[0].adversarial_review
    assert review.rejection_codes == ("alternative_grounded_path",)


def test_filter_validated_bundle_removes_adversarial_rejection(tmp_path):
    bundle = make_bundle("#!/usr/bin/env bash\necho PASSED")
    checks = dict.fromkeys(ADVERSARIAL_REVIEW_CHECKS, True)
    checks["single_lookup_resistant"] = False
    record = replace(
        bundle.questions[0],
        probe_validation=ProbeValidation(
            accepted_passed=True,
            incorrect_rejected=True,
        ),
        adversarial_review=AdversarialReview(
            accepted=False,
            checks=checks,
            reasons=("Answerable by one lookup.",),
        ),
    )
    bundle = replace(bundle, questions=(record,))
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "filtered"
    write_qa_bundle(bundle, bundle_dir)

    summary = filter_validated_bundle(bundle_dir, output_dir)

    assert summary["n_valid"] == 0
    assert not output_dir.exists()


def test_qa_cli_ingest_validate_and_evaluate_commands(tmp_path):
    runner = CliRunner()
    repository = tmp_path / "repository"
    make_repository(repository)
    generation_path = tmp_path / "generation.json"
    generation = {
        "instance_id": "repo__issue",
        "documentation": "The value file contains four.",
        "facts": [{"fact_id": "fact-001", "fact": "The value file contains four."}],
        "questions": [
            {
                "fact_id": "fact-001",
                "question_id": "q-001",
                "question": "What value is stored?",
                "accepted_answers": ["4"],
                "incorrect_answers": ["1", "2", "Error"],
                "answer_contract": {"type": "exact"},
                "category": "why_purpose_exploration",
                "evidence_hops": [
                    {
                        "evidence_id": "e1",
                        "file": "value.txt",
                        "symbol": "value",
                        "claim": "The file stores the value.",
                    },
                    {
                        "evidence_id": "e2",
                        "file": "reader.py",
                        "symbol": "read_value",
                        "claim": "The reader loads the file.",
                    },
                    {
                        "evidence_id": "e3",
                        "file": "service.py",
                        "symbol": "service",
                        "claim": "The service returns the loaded value.",
                    },
                ],
                "causal_chain": [
                    {
                        "step_id": "s1",
                        "claim": "The value is stored.",
                        "evidence_ids": ["e1"],
                    },
                    {
                        "step_id": "s2",
                        "claim": "The reader loads it.",
                        "evidence_ids": ["e2"],
                    },
                    {
                        "step_id": "s3",
                        "claim": "The service returns it.",
                        "evidence_ids": ["e3"],
                    },
                ],
                "probe": (
                    "#!/usr/bin/env bash\n"
                    "result=$(cat value.txt)\n"
                    "[[ \"$1\" == \"$result\" ]] && echo PASSED || echo FAILED"
                ),
            }
        ],
    }
    generation_path.write_text(json.dumps(generation))
    bundle_dir = tmp_path / "bundle"

    ingest = runner.invoke(
        app,
        ["qa-ingest", "--bundle", str(generation_path), "--output-dir", str(bundle_dir)],
    )
    validation = runner.invoke(
        app,
        [
            "qa-validate",
            "--bundle-dir",
            str(bundle_dir),
            "--repository",
            str(repository),
            "--timeout",
            "2",
        ],
    )
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps([{"question_id": "q-001", "answer": "4"}]))
    report_dir = tmp_path / "report"
    evaluation = runner.invoke(
        app,
        [
            "evaluate",
            "--bundle-dir",
            str(bundle_dir),
            "--agent-results",
            str(answers),
            "--output-dir",
            str(report_dir),
        ],
    )

    assert ingest.exit_code == 0
    assert validation.exit_code == 0
    assert evaluation.exit_code == 0
    assert (report_dir / "evaluation.md").exists()
