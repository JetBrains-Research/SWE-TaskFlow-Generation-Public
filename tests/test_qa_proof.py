import hashlib
import inspect
import json

import pytest

from interactive_benchmark_generation.qa_evaluation import (
    _phase_metrics,
    evaluate_agent_results,
    load_agent_results,
)
from interactive_benchmark_generation.qa_io import (
    load_generation_bundle,
    validate_golden_proofs,
    write_qa_bundle,
)
from interactive_benchmark_generation.qa_judge import (
    judge_answer_submission,
    judge_bundle_answers,
    judge_bundle_results,
    judge_proof_submission,
)
from interactive_benchmark_generation.qa_judge_prompts import (
    QA_ANSWER_JUDGE_SYSTEM_PROMPT,
    QA_JUDGE_SYSTEM_PROMPT,
    build_qa_answer_judge_prompt,
)
from interactive_benchmark_generation.qa_models import (
    AnswerJudgeAssessment,
    ProofExecutionRecord,
    ProofJudgeAssessment,
    QARecord,
)
from interactive_benchmark_generation.qa_validation import (
    ADVERSARIAL_REVIEW_CHECKS_V2,
    filter_validated_bundle,
    import_adversarial_review,
    import_answer_judge_report,
    import_golden_proof_execution_report,
    import_judge_report,
    import_proof_execution_report,
)


def proof_script(text):
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"\n'
        "PROOF_REQUIRED_TOOLS=(bash)\n"
        'for tool in "${PROOF_REQUIRED_TOOLS[@]}"; do\n'
        '    command -v "$tool" >/dev/null 2>&1 || {\n'
        "        printf 'missing required proof tool: %s\\n' \"$tool\" >&2\n"
        "        exit 2\n"
        "    }\n"
        "done\n"
        '[[ -f "$REPO_ROOT/pyproject.toml" ]] || {\n'
        "    printf 'missing repository source: pyproject.toml\\n' >&2\n"
        "    exit 2\n"
        "}\n"
        f"printf '%s\\n' '{text}'\n"
    )


def proof_metadata(instance_id, question_id, script):
    return {
        "path": f"golden-proofs/{instance_id}/{question_id}.sh",
        "format": "bash",
        "sha256": hashlib.sha256(script.encode()).hexdigest(),
        "timeout_seconds": 30,
        "max_output_bytes": 65536,
    }


def v2_bundle_value(question_ids=("q-001",), include_validation=False):
    instance_id = "repo__issue"
    facts = []
    questions = []
    golden_proofs = {}
    for index, question_id in enumerate(question_ids, start=1):
        script = proof_script(question_id)
        proof = proof_metadata(instance_id, question_id, script)
        facts.append(
            {
                "fact_id": f"fact-{index:03d}",
                "fact": f"Reference fact {index}.",
            }
        )
        questions.append(
            {
                "fact_id": f"fact-{index:03d}",
                "question_id": question_id,
                "question": f"Why does behavior {index} occur?",
                "category": "why_design_rationale",
                "golden_answer": {"canonical": f"Reference answer {index}."},
                "golden_proof": proof,
                "evaluation_mode": "proof_judged_v1",
                "evidence_hops": [
                    {
                        "evidence_id": "e1",
                        "file": "entry.py",
                        "symbol": "entry",
                        "claim": "The flow begins.",
                    },
                    {
                        "evidence_id": "e2",
                        "file": "flow.py",
                        "symbol": "middle",
                        "claim": "The flow transforms data.",
                    },
                    {
                        "evidence_id": "e3",
                        "file": "flow.py",
                        "symbol": "exit",
                        "claim": "The flow returns data.",
                    },
                ],
                "reasoning_chain": [
                    {
                        "step_id": "s1",
                        "claim": "The flow begins.",
                        "evidence_ids": ["e1"],
                    },
                    {
                        "step_id": "s2",
                        "claim": "Data is transformed.",
                        "evidence_ids": ["e2"],
                    },
                    {
                        "step_id": "s3",
                        "claim": "Data is returned.",
                        "evidence_ids": ["e3"],
                    },
                ],
            }
        )
        if include_validation:
            questions[-1]["golden_proof_validation"] = {
                "operationally_valid": True,
                "repository_dependent": True,
                "deterministic": True,
                "repository_state_unchanged": True,
            }
        golden_proofs[proof["path"]] = script
    return {
        "schema_version": "2.0",
        "instance_id": instance_id,
        "documentation": "Repository documentation.",
        "facts": facts,
        "questions": questions,
        "golden_proofs": golden_proofs,
    }


def test_raw_v2_record_serialization_omits_proof_validation():
    bundle = v2_bundle_value()
    value = {
        **bundle["questions"][0],
        "fact": bundle["facts"][0]["fact"],
        "schema_version": bundle["schema_version"],
    }
    record = QARecord.from_dict(
        value,
        default_instance_id=bundle["instance_id"],
    )

    assert "golden_proof_validation" not in record.to_dict()


def test_judge_requires_runtime_evidence_for_behavioral_claims():
    assert "AST parsing, regex matching, and source inspection" in QA_JUDGE_SYSTEM_PROMPT
    assert "insufficient as the primary evidence for behavior or causality" in (
        QA_JUDGE_SYSTEM_PROMPT
    )
    assert "Operational success does not establish behavioral truth" in (
        QA_JUDGE_SYSTEM_PROMPT
    )


def test_judge_defaults_to_sol_with_medium_reasoning():
    for function in (
        judge_proof_submission,
        judge_answer_submission,
        judge_bundle_results,
        judge_bundle_answers,
    ):
        parameters = inspect.signature(function).parameters
        assert parameters["model"].default == "openai/gpt-5.6-sol"
        assert parameters["reasoning_effort"].default == "medium"


def test_answer_judge_protocol_has_no_agent_proof_surface():
    payload = json.loads(
        build_qa_answer_judge_prompt(
            question="Why?",
            golden_answer="Because.",
            golden_proof_source="exit 0",
            golden_execution=execution(
                "q-001",
                "golden",
                "0" * 64,
                repository_dependent=True,
                deterministic=True,
            ),
            agent_answer="Because.",
            question_id="q-001",
            failure_codes=["agent_answer_incorrect"],
            checkpoint_context={"swe": {"reward": 1}},
        )
    )

    assert payload["protocol"] == "qa_answer_judge_input/v1"
    assert set(payload["agent_submission"]) == {"frozen_answer"}
    assert "never select contradicts_golden_but_agent_proof_wins" in (
        QA_ANSWER_JUDGE_SYSTEM_PROMPT
    )


def execution(
    question_id,
    proof_kind,
    digest,
    success=True,
    repository_dependent=None,
    deterministic=None,
):
    return {
        "protocol": "qa_proof_execution/v1",
        "question_id": question_id,
        "proof_kind": proof_kind,
        "source_sha256": digest,
        "submitted": True,
        "executed": True,
        "success": success,
        "exit_code": 0 if success else 1,
        "timed_out": False,
        "output_truncated": False,
        "repository_state_unchanged": True,
        "evidence_valid": True,
        "stdout": "evidence\n",
        "stderr": "",
        "duration_seconds": 0.1,
        **(
            {"repository_dependent": repository_dependent}
            if repository_dependent is not None
            else {}
        ),
        **({"deterministic": deterministic} if deterministic is not None else {}),
    }


def assessment(question_id, relation="same_as_golden"):
    dataset_issue = relation == "dataset_issue"
    alternative = relation in {
        "valid_extension_of_golden",
        "valid_alternative_to_golden",
        "contradicts_golden_but_agent_proof_wins",
    }
    return {
        "protocol": "qa_proof_judge/v1",
        "question_id": question_id,
        "agent_answer_correct": "yes",
        "agent_proof_matches_answer": "yes",
        "agent_proof_establishes_answer": "yes",
        "golden_proof_supports_golden_answer": "no" if dataset_issue else "yes",
        "gold_reference_status": "incomplete" if dataset_issue else "sufficient",
        "valid_alternative_to_golden": alternative,
        "relation": relation,
        "confidence": 0.9,
        "covered_claims": ["The behavior is repository-grounded."],
        "unsupported_claims": [],
        "contradicted_claims": [],
        "failure_codes": ["gold_reference_incomplete"] if dataset_issue else [],
        "rationale": "Trusted execution and source support the verdict.",
    }


def answer_assessment(question_id, relation="same_as_golden"):
    value = assessment(question_id, relation)
    value["protocol"] = "qa_answer_judge/v1"
    value.pop("agent_proof_matches_answer")
    value.pop("agent_proof_establishes_answer")
    return value


def test_proof_execution_accepts_explicit_invalid_evidence():
    value = execution("q-001", "agent", "0" * 64)
    value["success"] = False
    value["evidence_valid"] = False

    record = ProofExecutionRecord.from_dict(value)

    assert record.success is False
    assert record.evidence_valid is False


def test_proof_execution_rejects_success_with_invalid_evidence():
    value = execution("q-001", "agent", "0" * 64)
    value["evidence_valid"] = False

    with pytest.raises(ValueError, match="trusted execution integrity"):
        ProofExecutionRecord.from_dict(value)


def test_proof_execution_requires_explicit_evidence_validity():
    value = execution("q-001", "agent", "0" * 64)
    del value["evidence_valid"]

    with pytest.raises(ValueError, match="missing keys"):
        ProofExecutionRecord.from_dict(value)


def test_v2_bundle_round_trip_checks_safe_paths_hash_and_executable(tmp_path):
    bundle = load_generation_bundle_value(v2_bundle_value())
    output = tmp_path / "bundle"

    write_qa_bundle(bundle, output)

    assert bundle.questions[0].golden_proof_validation is None
    assert load_generation_bundle(output).questions[0].golden_proof_validation is None
    report = validate_golden_proofs(load_generation_bundle(output), output)
    assert report["n_valid"] == 1
    assert report["proofs"][0]["required_tools"] == ["bash"]
    assert report["proofs"][0]["asserted_source_files"] == ["pyproject.toml"]
    assert report["proofs"][0]["repository_write_commands"] == []
    assert (output / bundle.questions[0].golden_proof.path).stat().st_mode & 0o111

    unsafe = v2_bundle_value()
    unsafe["questions"][0]["golden_proof"]["path"] = "../proof.sh"
    with pytest.raises(ValueError, match="safe relative path"):
        load_generation_bundle_value(unsafe)

    bad_hash = v2_bundle_value()
    bad_hash["questions"][0]["golden_proof"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        write_qa_bundle(load_generation_bundle_value(bad_hash), tmp_path / "bad")


def test_golden_proof_preflight_allows_temporary_file_cleanup(tmp_path):
    script = proof_script("q-001").replace(
        "printf '%s\\n' 'q-001'\n",
        'temporary=$(mktemp -d)\n'
        "trap 'rm -rf \"$temporary\"' EXIT\n"
        'mkdir -p "$temporary/nested"\n'
        'touch "$temporary/nested/evidence"\n'
        "printf '%s\\n' 'q-001'\n",
    )
    value = v2_bundle_value()
    proof = value["questions"][0]["golden_proof"]
    proof["sha256"] = hashlib.sha256(script.encode()).hexdigest()
    value["golden_proofs"][proof["path"]] = script

    write_qa_bundle(
        load_generation_bundle_value(value),
        tmp_path / "bundle",
    )


def test_golden_proof_preflight_accepts_literal_source_array_loop(tmp_path):
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'REPO_ROOT="${REPO_ROOT:?required}"\n'
        "PROOF_REQUIRED_TOOLS=(python3)\n"
        'for tool in "${PROOF_REQUIRED_TOOLS[@]}"; do\n'
        '    command -v "$tool" >/dev/null 2>&1 || {\n'
        "        printf 'missing required proof tool: %s\\n' \"$tool\" >&2\n"
        "        exit 2\n"
        "    }\n"
        "done\n"
        'EXPECTED_SOURCE_FILES=("src/package.py" "src/runtime.py")\n'
        'for source_file in "${EXPECTED_SOURCE_FILES[@]}"; do\n'
        '    if [[ ! -f "$REPO_ROOT/$source_file" ]]; then\n'
        "        exit 2\n"
        "    fi\n"
        "done\n"
        'export PYTHONPATH="$REPO_ROOT/src"\n'
        "export PYTHONNOUSERSITE=1\n"
        "export PYTHONSAFEPATH=1\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        'cd "$REPO_ROOT"\n'
        "python3 -c 'print(1)'\n"
    )
    value = v2_bundle_value()
    proof = value["questions"][0]["golden_proof"]
    proof["sha256"] = hashlib.sha256(script.encode()).hexdigest()
    value["golden_proofs"][proof["path"]] = script

    output = tmp_path / "bundle"
    write_qa_bundle(load_generation_bundle_value(value), output)

    report = validate_golden_proofs(load_generation_bundle(output), output)
    assert report["proofs"][0]["asserted_source_files"] == [
        "src/package.py",
        "src/runtime.py",
    ]


@pytest.mark.parametrize(
    ("script", "diagnostic"),
    [
        (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'REPO_ROOT="${REPO_ROOT:?required}"\n'
            "PROOF_REQUIRED_TOOLS=(bash)\n"
            "command -v bash || printf 'missing required proof tool\\n' >&2\n"
            '[[ -f "$REPO_ROOT/.git/HEAD" ]]\n',
            "exact source file under REPO_ROOT",
        ),
        (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'REPO_ROOT="${REPO_ROOT:?required}"\n'
            "PROOF_REQUIRED_TOOLS=(python3)\n"
            "command -v python3 || printf 'missing required proof tool\\n' >&2\n"
            '[[ -f "$REPO_ROOT/src/package.py" ]]\n'
            "python3 -c 'import package'\n",
            "Python proof must configure PYTHONPATH",
        ),
        (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'REPO_ROOT="${REPO_ROOT:?required}"\n'
            "PROOF_REQUIRED_TOOLS=(python3)\n"
            "command -v python3 || printf 'missing required proof tool\\n' >&2\n"
            'PYTHONPATH="$REPO_ROOT"\n'
            "PYTHONNOUSERSITE=1\n"
            "PYTHONSAFEPATH=1\n"
            "PYTHONDONTWRITEBYTECODE=1\n"
            "python3 -c 'import package'\n"
            '[[ -f "$REPO_ROOT/src/package.py" ]]\n'
            'cd "$REPO_ROOT"\n',
            "assert repository source before running Python",
        ),
        (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'REPO_ROOT="${REPO_ROOT:?required}"\n'
            "PROOF_REQUIRED_TOOLS=(bash)\n"
            "command -v bash || printf 'missing required proof tool\\n' >&2\n"
            '[[ -f "$REPO_ROOT/src/package.py" ]]\n'
            'touch "$REPO_ROOT/result.txt"\n',
            "may write repository state: touch",
        ),
    ],
)
def test_golden_proof_structural_preflight_reports_unsafe_source(
    tmp_path,
    script,
    diagnostic,
):
    value = v2_bundle_value()
    proof = value["questions"][0]["golden_proof"]
    proof["sha256"] = hashlib.sha256(script.encode()).hexdigest()
    value["golden_proofs"][proof["path"]] = script

    with pytest.raises(ValueError, match=diagnostic):
        write_qa_bundle(
            load_generation_bundle_value(value),
            tmp_path / "bundle",
        )


def test_golden_only_import_attaches_validation_and_enables_filtering(tmp_path):
    bundle = load_generation_bundle_value(v2_bundle_value())
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    record = bundle.questions[0]
    golden_report = tmp_path / "golden.json"
    golden_report.write_text(
        json.dumps(
            {
                "executions": [
                    execution(
                        "q-001",
                        "golden",
                        record.golden_proof.sha256,
                        repository_dependent=True,
                        deterministic=True,
                    )
                ]
            }
        )
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "instance_id": "repo__issue",
                "reviews": [
                    {
                        "question_id": "q-001",
                        "accepted": True,
                        "checks": dict.fromkeys(
                            ADVERSARIAL_REVIEW_CHECKS_V2,
                            True,
                        ),
                        "reasons": [],
                    }
                ],
            }
        )
    )

    imported = import_golden_proof_execution_report(bundle_dir, golden_report)
    import_adversarial_review(bundle_dir, review_path)
    filtered_dir = tmp_path / "filtered"
    summary = filter_validated_bundle(bundle_dir, filtered_dir)
    filtered = load_generation_bundle(filtered_dir)

    validation = load_generation_bundle(bundle_dir).questions[
        0
    ].golden_proof_validation
    assert imported["stage"] == "golden"
    assert validation.operationally_valid is True
    assert validation.repository_dependent is True
    assert validation.deterministic is True
    assert summary["n_valid"] == 1
    assert filtered.schema_version == "2.0"
    assert filtered.questions[0].golden_proof_validation == validation
    assert set(filtered.golden_proofs) == {record.golden_proof.path}
    assert (
        filtered.questions[0]
        .adversarial_review.checks["alternative_answers_accounted_for"]
        is True
    )
    filtered_golden_report = json.loads(
        (filtered_dir / "golden-proof-executions.json").read_text()
    )
    assert filtered_golden_report["stage"] == "golden"
    assert filtered_golden_report["n_executions"] == 1
    assert {
        item["proof_kind"]
        for item in filtered_golden_report["executions"]
    } == {"golden"}
    agent_report = tmp_path / "agent.json"
    agent_report.write_text(
        json.dumps({
            "executions": [
                execution("q-001", "agent", "1" * 64),
            ]
        })
    )

    combined = import_proof_execution_report(
        filtered_dir,
        agent_report,
        stage="agent",
    )

    assert combined["stage"] == "combined"
    assert combined["n_executions"] == 2
    assert {
        item["proof_kind"]
        for item in combined["executions"]
    } == {"golden", "agent"}


def test_v2_review_requires_and_agrees_with_golden_validation(tmp_path):
    bundle = load_generation_bundle_value(v2_bundle_value())
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    checks = dict.fromkeys(ADVERSARIAL_REVIEW_CHECKS_V2, True)
    report_path = tmp_path / "review.json"
    report = {
        "instance_id": "repo__issue",
        "reviews": [
            {
                "question_id": "q-001",
                "accepted": True,
                "checks": checks,
                "reasons": [],
            }
        ],
    }
    report_path.write_text(json.dumps(report))

    with pytest.raises(
        ValueError,
        match="requires attached golden proof validation",
    ):
        import_adversarial_review(bundle_dir, report_path)

    record = bundle.questions[0]
    execution_path = tmp_path / "golden.json"
    execution_path.write_text(
        json.dumps(
            {
                "executions": [
                    execution(
                        "q-001",
                        "golden",
                        record.golden_proof.sha256,
                        repository_dependent=True,
                        deterministic=True,
                    )
                ]
            }
        )
    )
    import_golden_proof_execution_report(bundle_dir, execution_path)
    report["reviews"][0]["checks"]["golden_proof_operationally_valid"] = False
    report["reviews"][0]["accepted"] = False
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="disagrees with trusted validation"):
        import_adversarial_review(bundle_dir, report_path)


def test_v2_review_rejects_trusted_false_reviewer_true_mismatch(tmp_path):
    bundle = load_generation_bundle_value(v2_bundle_value())
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    golden_report = tmp_path / "golden.json"
    golden_report.write_text(
        json.dumps(
            {
                "executions": [
                    execution(
                        "q-001",
                        "golden",
                        bundle.questions[0].golden_proof.sha256,
                        success=False,
                        repository_dependent=True,
                        deterministic=True,
                    )
                ]
            }
        )
    )
    import_golden_proof_execution_report(bundle_dir, golden_report)
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "instance_id": "repo__issue",
                "reviews": [
                    {
                        "question_id": "q-001",
                        "accepted": True,
                        "checks": dict.fromkeys(
                            ADVERSARIAL_REVIEW_CHECKS_V2,
                            True,
                        ),
                        "reasons": [],
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="disagrees with trusted validation"):
        import_adversarial_review(bundle_dir, review_path)


def test_adversarial_review_rejects_mixed_record_versions(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    v2_record = load_generation_bundle_value(v2_bundle_value()).questions[0]
    v1_record = {
        "schema_version": "1.0",
        "instance_id": "repo__issue",
        "fact_id": "fact-v1",
        "fact": "Legacy fact.",
        "question_id": "q-v1",
        "question": "What is the legacy value?",
        "accepted_answers": ["value"],
        "incorrect_answers": ["one", "two", "three"],
        "probe_path": "probes/repo__issue/q-v1.sh",
    }
    (bundle_dir / "questions.jsonl").write_text(
        json.dumps(v1_record) + "\n" + json.dumps(v2_record.to_dict()) + "\n"
    )
    report_path = tmp_path / "review.json"
    report_path.write_text(
        json.dumps({"instance_id": "repo__issue", "reviews": []})
    )

    with pytest.raises(ValueError, match="mixed schema versions"):
        import_adversarial_review(bundle_dir, report_path)


def test_agent_stage_merges_with_prior_golden_import(tmp_path):
    bundle = load_generation_bundle_value(v2_bundle_value())
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    record = bundle.questions[0]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "executions": [
                    execution(
                        "q-001",
                        "golden",
                        record.golden_proof.sha256,
                        repository_dependent=True,
                        deterministic=True,
                    )
                ]
            }
        )
    )
    agent_script = proof_script("agent")
    agent_path = tmp_path / "agent.json"
    agent_path.write_text(
        json.dumps(
            {
                "executions": [
                    execution(
                        "q-001",
                        "agent",
                        hashlib.sha256(agent_script.encode()).hexdigest(),
                    )
                ]
            }
        )
    )

    import_proof_execution_report(bundle_dir, golden_path, stage="golden")
    imported = import_proof_execution_report(
        bundle_dir,
        agent_path,
        stage="agent",
    )

    assert imported["stage"] == "combined"
    assert {
        execution_value["proof_kind"]
        for execution_value in imported["executions"]
    } == {"golden", "agent"}


def load_generation_bundle_value(value):
    from interactive_benchmark_generation.qa_models import GenerationBundle

    return GenerationBundle.from_dict(value)


def test_judge_accepts_stronger_alternative_and_retries_malformed(monkeypatch):
    record = load_generation_bundle_value(v2_bundle_value()).questions[0]
    agent_script = proof_script("alternative")
    digest = hashlib.sha256(agent_script.encode()).hexdigest()
    golden = ProofExecutionRecord.from_dict(
        execution("q-001", "golden", record.golden_proof.sha256)
    )
    agent = ProofExecutionRecord.from_dict(
        execution("q-001", "agent", digest)
    )
    responses = [
        {"protocol": "wrong"},
        assessment("q-001", "valid_alternative_to_golden"),
    ]

    monkeypatch.setattr(
        "interactive_benchmark_generation.qa_judge.call_json_model",
        lambda *args, **kwargs: responses.pop(0),
    )
    result = judge_proof_submission(
        record,
        proof_script("gold"),
        golden,
        "A stronger alternative.",
        agent_script,
        agent,
        max_attempts=2,
        retry_delay=0,
    )

    assert result.answer_correct is True
    assert result.valid_alternative_to_golden is True
    assert result.answer_plus_proof_correct(agent) is True


def test_judge_skips_model_when_trusted_golden_execution_failed(
    tmp_path,
    monkeypatch,
):
    bundle = load_generation_bundle_value(v2_bundle_value())
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    results_path = tmp_path / "answers.json"
    results_path.write_text(
        json.dumps([{"question_id": "q-001", "answer": "Answer."}])
    )
    executions_path = tmp_path / "executions.json"
    executions_path.write_text(
        json.dumps(
            {
                "executions": [
                    execution(
                        "q-001",
                        "golden",
                        bundle.questions[0].golden_proof.sha256,
                        success=False,
                        repository_dependent=True,
                        deterministic=True,
                    ),
                    execution("q-001", "agent", "a" * 64),
                ]
            }
        )
    )
    monkeypatch.setattr(
        "interactive_benchmark_generation.qa_judge.call_json_model",
        lambda *args, **kwargs: pytest.fail("judge model must not be called"),
    )

    report = judge_bundle_results(
        bundle_dir,
        results_path,
        executions_path,
        tmp_path / "judge.json",
    )

    assert report["assessments"] == []
    assert "trusted golden proof is not operationally valid" in (
        report["errors"][0]["error"]
    )


def test_judge_schema_rejects_malformed_or_unknown_values():
    value = assessment("q-001")
    value["relation"] = "close_enough"
    with pytest.raises(ValueError, match="close_enough"):
        ProofJudgeAssessment.from_dict(value)

    value = assessment("q-001")
    value["unexpected"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        ProofJudgeAssessment.from_dict(value)


def test_answer_judge_rejects_proof_winning_relation():
    value = answer_assessment(
        "q-001",
        "contradicts_golden_but_agent_proof_wins",
    )

    with pytest.raises(ValueError, match="proof-winning relation"):
        AnswerJudgeAssessment.from_dict(value)


def test_judge_import_normalizes_valid_alternative_from_relation():
    value = assessment("q-001", "valid_extension_of_golden")
    value["valid_alternative_to_golden"] = False

    result = ProofJudgeAssessment.from_dict(value)

    assert result.valid_alternative_to_golden is True
    assert result.relation.value == "valid_extension_of_golden"


def test_answer_only_import_and_evaluation_mark_proof_not_applicable(
    tmp_path,
):
    bundle = load_generation_bundle_value(v2_bundle_value())
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        json.dumps(
            {
                "question_id": "q-001",
                "answer": "Agent answer.",
                "answer_phase": {"cost": 0.1, "steps": 2},
                "proof_phase": None,
            }
        )
        + "\n"
    )
    golden_report = tmp_path / "golden-executions.json"
    golden_report.write_text(
        json.dumps(
            {
                "protocol": "qa_proof_execution_report/v1",
                "executions": [
                    execution(
                        "q-001",
                        "golden",
                        bundle.questions[0].golden_proof.sha256,
                        repository_dependent=True,
                        deterministic=True,
                    )
                ],
            }
        )
    )
    judge_path = tmp_path / "answer-judge.json"
    judge_path.write_text(
        json.dumps(
            {
                "protocol": "qa_answer_judge_report/v1",
                "model": "openai/gpt-5.6-sol",
                "reasoning_effort": "medium",
                "cost": 0.2,
                "assessments": [answer_assessment("q-001")],
                "errors": [],
            }
        )
    )

    imported = import_answer_judge_report(bundle_dir, judge_path)
    report = evaluate_agent_results(
        bundle_dir,
        answers,
        tmp_path / "answer-report",
        proof_executions_path=golden_report,
        judge_report_path=bundle_dir
        / "answer-judge-assessments.json",
    )

    assert imported["protocol"] == "qa_answer_judge_report/v1"
    assert report["evaluation_mode"] == "answer_judged_v1"
    assert report["n_answer_correct"] == 1
    assert report["answer_plus_proof_accuracy"] is None
    assert report["questions"][0]["answer_plus_proof_correct"] is None
    assert report["phase_metrics"]["proof"]["steps"] == 0
    assert report["phase_metrics"]["proof"]["cost"] == 0


def test_import_and_evaluation_exclude_dataset_issue_and_report_rates(tmp_path):
    bundle = load_generation_bundle_value(v2_bundle_value(("q-001", "q-002")))
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    agent_scripts = {
        question_id: proof_script(f"agent-{question_id}")
        for question_id in ("q-001", "q-002")
    }
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "answer": f"Agent answer {question_id}.",
                    "proof_script": script,
                    "answer_metrics": {"cost": 0.1, "steps": 2},
                    "proof_metrics": {"cost": 0.2, "steps": 3},
                }
            )
            + "\n"
            for question_id, script in agent_scripts.items()
        )
    )
    executions = []
    for record in bundle.questions:
        executions.extend(
            [
                execution(
                    record.question_id,
                    "golden",
                    record.golden_proof.sha256,
                ),
                execution(
                    record.question_id,
                    "agent",
                    hashlib.sha256(
                        agent_scripts[record.question_id].encode()
                    ).hexdigest(),
                ),
            ]
        )
    execution_path = tmp_path / "executions.json"
    execution_path.write_text(json.dumps({"executions": executions}))
    judge_path = tmp_path / "judge.json"
    judge_path.write_text(
        json.dumps(
            {
                "model": "openai/gpt-5.6-sol",
                "reasoning_effort": "medium",
                "cost": 0.5,
                "assessments": [
                    assessment("q-001", "valid_extension_of_golden"),
                    assessment("q-002", "dataset_issue"),
                ],
            }
        )
    )

    imported_executions = import_proof_execution_report(
        bundle_dir,
        execution_path,
    )
    imported_judge = import_judge_report(bundle_dir, judge_path)
    report = evaluate_agent_results(
        bundle_dir,
        answers,
        tmp_path / "report",
        judge_cost_override=0.75,
    )

    assert imported_executions["n_executions"] == 4
    assert len(imported_judge["assessments"]) == 2
    assert report["n_questions"] == 2
    assert report["n_scored"] == 1
    assert report["n_dataset_issues"] == 1
    assert report["answer_accuracy"] == 1
    assert report["answer_plus_proof_accuracy"] == 1
    assert report["valid_alternative_rate"] == 1
    assert report["proof_submission_rate"] == 1
    assert report["proof_execution_rate"] == 1
    assert report["phase_metrics"]["answer"]["cost"] == pytest.approx(0.2)
    assert report["phase_metrics"]["proof"]["cost"] == pytest.approx(0.4)
    assert report["average_answer_steps"] == 2
    assert report["average_proof_steps"] == 3
    assert report["average_steps_per_answer"] == 5
    assert report["judge_cost"] == 0.75
    assert "Average steps per answer: 5.00" in (
        tmp_path / "report" / "evaluation.md"
    ).read_text()


@pytest.mark.parametrize("include_proof", [False, True])
def test_correct_answer_with_missing_or_failed_proof_fails_combined_metric(
    tmp_path,
    include_proof,
):
    bundle = load_generation_bundle_value(v2_bundle_value())
    bundle_dir = tmp_path / "bundle"
    write_qa_bundle(bundle, bundle_dir)
    agent_script = proof_script("agent")
    digest = hashlib.sha256(agent_script.encode()).hexdigest()
    result = {"question_id": "q-001", "answer": "Correct answer."}
    if include_proof:
        result["proof"] = {"script": agent_script, "sha256": digest}
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps([result]))
    executions = {
        "executions": [
            execution(
                "q-001",
                "golden",
                bundle.questions[0].golden_proof.sha256,
            ),
            execution("q-001", "agent", digest, success=False),
        ]
    }
    (bundle_dir / "proof-executions.json").write_text(json.dumps(executions))
    (bundle_dir / "judge-assessments.json").write_text(
        json.dumps({"assessments": [assessment("q-001")]})
    )

    report = evaluate_agent_results(bundle_dir, answers, tmp_path / "report")

    question = report["questions"][0]
    assert question["answer_correct"] is True
    assert question["answer_plus_proof_correct"] is False
    assert report["answer_correct_proof_failed"] == 1
    assert (
        "agent_proof_execution_failed" in question["failures"]
        if include_proof
        else "missing_agent_proof" in question["failures"]
    )


def test_directory_results_materialize_and_verify_proof_sidecars(tmp_path):
    result_dir = tmp_path / "results"
    proof_path = result_dir / "agent-proofs/repo__issue/q-001.sh"
    proof_path.parent.mkdir(parents=True)
    script = proof_script("agent")
    proof_path.write_text(script)
    digest = hashlib.sha256(script.encode()).hexdigest()
    (result_dir / "answers.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q-001",
                "frozen_answer": "Answer.",
                "answer_phase": {
                    "steps": 2,
                    "cost": 0.1,
                    "usage_records": [
                        {
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "prompt_tokens_details": {"cached_tokens": 4},
                            }
                        }
                    ],
                },
            }
        )
        + "\n"
    )
    proof_record = {
        "question_id": "q-001",
        "path": "agent-proofs/repo__issue/q-001.sh",
        "sha256": digest,
        "status": "submitted",
        "proof_phase": {
            "steps": 3,
            "cost": 0.2,
            "usage_records": [
                {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 8},
                    }
                }
            ],
        },
    }
    (result_dir / "proofs.jsonl").write_text(json.dumps(proof_record) + "\n")

    loaded = load_agent_results(result_dir)

    assert loaded[0]["answer"] == "Answer."
    assert loaded[0]["proof"]["script"] == script
    assert "artifact_error" not in loaded[0]["proof"]
    assert _phase_metrics(loaded[0], "answer") == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cached_tokens": 4,
        "cost": 0.1,
        "steps": 2,
        "tool_calls": 0,
        "latency_seconds": 0,
    }
    assert _phase_metrics(loaded[0], "proof") == {
        "input_tokens": 20,
        "output_tokens": 5,
        "cached_tokens": 8,
        "cost": 0.2,
        "steps": 3,
        "tool_calls": 0,
        "latency_seconds": 0,
    }

    proof_record["sha256"] = "0" * 64
    (result_dir / "proofs.jsonl").write_text(json.dumps(proof_record) + "\n")
    mismatched = load_agent_results(result_dir)

    assert mismatched[0]["proof"]["status"] == "malformed"
    assert mismatched[0]["proof"]["artifact_error"] == "proof_hash_mismatch"
    assert "script" not in mismatched[0]["proof"]


@pytest.mark.parametrize(
    ("path_value", "error"),
    [
        ("../proof.sh", "unsafe_proof_path"),
        ("agent-proofs/missing.sh", "missing_proof_file"),
    ],
)
def test_directory_results_reject_unsafe_or_missing_sidecar(
    tmp_path,
    path_value,
    error,
):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "answers.jsonl").write_text(
        json.dumps({"question_id": "q-001", "answer": "Answer."}) + "\n"
    )
    (result_dir / "proofs.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q-001",
                "path": path_value,
                "sha256": "0" * 64,
                "status": "submitted",
            }
        )
        + "\n"
    )

    loaded = load_agent_results(result_dir)

    assert loaded[0]["proof"]["artifact_error"] == error


def test_proof_aware_trajectory_prefers_question_records_and_phase_metrics(
    tmp_path,
):
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "info": {
                    "question_records": [
                        {
                            "question_id": "q-001",
                            "frozen_answer": "Frozen answer.",
                            "proof_script": proof_script("agent"),
                            "proof_sha256": "a" * 64,
                            "answer_phase": {
                                "metrics": {"cost": 0.1, "steps": 2}
                            },
                            "proof_phase": {
                                "metrics": {"cost": 0.2, "steps": 3}
                            },
                        }
                    ],
                    "answer_records": [
                        {"question_id": "q-001", "answer": "Stale answer."}
                    ],
                }
            }
        )
    )

    loaded = load_agent_results(trajectory)

    assert loaded[0]["answer"] == "Frozen answer."
    assert loaded[0]["proof"]["script"] == proof_script("agent")
    assert loaded[0]["answer_phase"]["metrics"]["steps"] == 2
    assert loaded[0]["proof_phase"]["metrics"]["steps"] == 3
    assert _phase_metrics(loaded[0], "answer")["cost"] == 0.1
    assert _phase_metrics(loaded[0], "proof")["cost"] == 0.2
