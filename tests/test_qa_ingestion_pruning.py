"""One question's defect must not take its issue down with it.

Ingestion used to raise on the first unusable question, so an issue lost every
question because one of them reused an evidence id or shipped a proof that failed
structural preflight. Standing policy is now: drop the offending question, keep the
rest, and fail only when nothing is left.

Two questions can also each be valid yet unable to coexist — a fact may be tested
only once. There the harder question wins, ranked by evidence hops, then cross-file
breadth, then chain length, then question id for reproducibility.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from interactive_benchmark_generation.qa_io import (
    _question_rank,
    load_generation_bundle,
    prune_unusable_questions,
)

PROOF = """#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:?required}"
PROOF_REQUIRED_TOOLS=(bash)
command -v bash >/dev/null || { printf 'missing required proof tool\\n' >&2; exit 2; }
[[ -f "$REPO_ROOT/src/package.py" ]] || exit 2
printf '%s\\n' '{"protocol":"qa_proof/v1","observation":"x","method":"m","evidence":{"a":1}}'
"""

INSTANCE = "instance_project__repo-v1"


def hop(index, file):
    return {
        "evidence_id": f"e{index}",
        "file": file,
        "symbol": f"symbol_{index}",
        "claim": f"claim from hop {index}",
    }


def step(index, hops):
    # Steps must cite evidence that exists, or the question is dropped as defective.
    return {
        "step_id": f"s{index}",
        "claim": f"claim {index}",
        "evidence_ids": [f"e{index % hops}"],
    }


def question(question_id, fact_id, *, hops=3, files=2, chain=3, proof=PROOF):
    files_cycle = [f"src/file{i % files}.py" for i in range(hops)]
    return {
        "instance_id": INSTANCE,
        "fact_id": fact_id,
        "question_id": question_id,
        "question": f"Why does {question_id} behave that way across components?",
        "category": "how_algorithm_implementation",
        "golden_answer": {"canonical": f"Because of {question_id}."},
        "evidence_hops": [hop(i, files_cycle[i]) for i in range(hops)],
        "reasoning_chain": [step(i, hops) for i in range(chain)],
        "evaluation_mode": "proof_judged_v1",
        "golden_proof": {
            "path": f"golden-proofs/{INSTANCE}/{question_id}.sh",
            "format": "bash",
            "sha256": hashlib.sha256(proof.encode()).hexdigest(),
            "timeout_seconds": 30,
            "max_output_bytes": 8192,
        },
    }


def bundle(*questions):
    facts = {q["fact_id"] for q in questions}
    return {
        "schema_version": "2.0",
        "instance_id": INSTANCE,
        "documentation": "Documentation for the repository.",
        "facts": [{"fact_id": f, "fact": f"Fact {f}", "instance_id": INSTANCE} for f in facts],
        "questions": list(questions),
        "probes": {},
        "golden_proofs": {q["golden_proof"]["path"]: PROOF for q in questions},
        "metadata": {},
        "debug": {},
    }


def kept_ids(value):
    pruned, _ = prune_unusable_questions(value)
    return [q["question_id"] for q in pruned["questions"]]


def test_a_healthy_bundle_is_untouched():
    value = bundle(question("q-1", "f-1"), question("q-2", "f-2"))
    pruned, dropped = prune_unusable_questions(value)
    assert [q["question_id"] for q in pruned["questions"]] == ["q-1", "q-2"]
    assert dropped == []
    assert "ingestion_dropped_questions" not in pruned["debug"]


def test_a_question_reusing_an_evidence_id_goes_alone():
    # The uniqueness rule is within one question, so this is that question's defect.
    broken = question("q-bad", "f-bad")
    broken["evidence_hops"][1]["evidence_id"] = broken["evidence_hops"][0]["evidence_id"]
    value = bundle(question("q-1", "f-1"), broken, question("q-2", "f-2"))

    pruned, dropped = prune_unusable_questions(value)

    assert [q["question_id"] for q in pruned["questions"]] == ["q-1", "q-2"]
    assert [d["question_id"] for d in dropped] == ["q-bad"]
    assert dropped[0]["reason"] == "question_defect"
    assert "Evidence identifiers must be unique" in dropped[0]["detail"]


def test_a_question_whose_proof_fails_preflight_goes_alone():
    bad_proof = PROOF.replace('[[ -f "$REPO_ROOT/src/package.py" ]] || exit 2\n', "")
    broken = question("q-bad", "f-bad", proof=bad_proof)
    value = bundle(question("q-1", "f-1"), broken)
    value["golden_proofs"][broken["golden_proof"]["path"]] = bad_proof

    pruned, dropped = prune_unusable_questions(value)

    assert [q["question_id"] for q in pruned["questions"]] == ["q-1"]
    assert "exact source file under REPO_ROOT" in dropped[0]["detail"]


def test_surviving_questions_keep_only_their_own_proofs():
    broken = question("q-bad", "f-bad")
    broken["evidence_hops"][1]["evidence_id"] = broken["evidence_hops"][0]["evidence_id"]
    value = bundle(question("q-1", "f-1"), broken)

    pruned, _ = prune_unusable_questions(value)

    assert set(pruned["golden_proofs"]) == {f"golden-proofs/{INSTANCE}/q-1.sh"}


def test_two_questions_on_one_fact_keep_the_harder():
    easy = question("q-easy", "f-shared", hops=3)
    hard = question("q-hard", "f-shared", hops=5)
    # Order in the bundle must not decide it.
    assert kept_ids(bundle(easy, hard)) == ["q-hard"]
    assert kept_ids(bundle(hard, easy)) == ["q-hard"]

    _, dropped = prune_unusable_questions(bundle(easy, hard))
    assert dropped[0]["question_id"] == "q-easy"
    assert dropped[0]["reason"] == "duplicate_fact"
    assert "kept the harder question" in dropped[0]["detail"]


def test_equal_hops_are_broken_by_cross_file_breadth_then_chain():
    narrow = question("q-narrow", "f-shared", hops=4, files=2, chain=3)
    wide = question("q-wide", "f-shared", hops=4, files=4, chain=3)
    assert kept_ids(bundle(narrow, wide)) == ["q-wide"]

    short = question("q-short", "f-shared", hops=4, files=2, chain=3)
    long = question("q-long", "f-shared", hops=4, files=2, chain=5)
    assert kept_ids(bundle(short, long)) == ["q-long"]


def test_a_full_tie_is_resolved_deterministically():
    a = question("q-aaa", "f-shared", hops=4)
    b = question("q-bbb", "f-shared", hops=4)
    assert kept_ids(bundle(a, b)) == kept_ids(bundle(b, a))


def test_rank_orders_hardest_first():
    ranks = [
        _question_rank(question("q", "f", hops=hops, files=files, chain=chain))
        for hops, files, chain in [(3, 2, 3), (4, 2, 3), (4, 4, 3), (4, 4, 5)]
    ]
    assert ranks == sorted(ranks)


def test_an_issue_dies_only_when_nothing_survives():
    broken = question("q-bad", "f-bad")
    broken["evidence_hops"][1]["evidence_id"] = broken["evidence_hops"][0]["evidence_id"]
    with pytest.raises(ValueError, match="No ingestible questions remain"):
        prune_unusable_questions(bundle(broken))


def test_ingestion_prunes_but_a_plain_load_stays_strict(tmp_path):
    broken = question("q-bad", "f-bad")
    broken["evidence_hops"][1]["evidence_id"] = broken["evidence_hops"][0]["evidence_id"]
    value = bundle(question("q-1", "f-1"), broken)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    loaded = load_generation_bundle(path, prune=True)
    assert [record.question_id for record in loaded.questions] == ["q-1"]
    assert loaded.debug["ingestion_dropped_questions"][0]["question_id"] == "q-bad"

    # Re-loading a written bundle must not silently discard corruption.
    with pytest.raises(ValueError, match="Evidence identifiers must be unique"):
        load_generation_bundle(path)
