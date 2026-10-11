#!/usr/bin/env python3
"""Materialize a SWE-TaskFlow issue+QA row as a validated bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def validate_row(row: dict) -> None:
    hybrid = json.loads(row["hybrid"])
    digest_input = dict(hybrid)
    digest = digest_input.pop("record_sha256")
    actual = hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest()
    if digest != actual or row["record_sha256"] != actual:
        raise ValueError("Hybrid record SHA-256 is invalid")
    questions = json.loads(row["questions"])
    scripts = json.loads(row["golden_proof_scripts"])
    report = json.loads(row["golden_proof_executions"])
    question_ids = [question["question_id"] for question in questions]
    execution_ids = [
        execution["question_id"]
        for execution in report["executions"]
        if execution.get("proof_kind") == "golden" and execution.get("success") is True
    ]
    if question_ids != json.loads(row["stable_question_ids"]):
        raise ValueError("Stable question IDs differ from question records")
    if set(scripts) != set(question_ids) or execution_ids != question_ids:
        raise ValueError("Question, proof, and execution references differ")


def proof_output_path(
    bundle_dir: Path,
    instance_id: str,
    question: dict,
) -> Path:
    proof = question.get("golden_proof")
    if not isinstance(proof, dict) or not isinstance(proof.get("path"), str):
        raise TypeError("Question has no golden proof path")
    relative = Path(proof["path"])
    expected_parent = Path("golden-proofs") / instance_id
    if relative.is_absolute() or ".." in relative.parts or relative.parent != expected_parent:
        raise ValueError("Golden proof path is outside the expected instance directory")
    return bundle_dir / relative


def materialize(row: dict, output: Path) -> Path:
    validate_row(row)
    instance_id = row["instance_id"]
    questions = json.loads(row["questions"])
    scripts = json.loads(row["golden_proof_scripts"])
    report = json.loads(row["golden_proof_executions"])
    facts = json.loads(row["facts"])
    bundle_dir = output / instance_id / "filtered"
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise ValueError(f"Output bundle is not empty: {bundle_dir}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(bundle_dir / "questions.jsonl", questions)
    (bundle_dir / "facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "documentation.md").write_text(
        row["documentation"],
        encoding="utf-8",
    )
    (bundle_dir / "golden-proof-executions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = json.loads(row["bundle_metadata"])
    if metadata:
        (bundle_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    questions_by_id = {question["question_id"]: question for question in questions}
    for question_id, source in scripts.items():
        path = proof_output_path(
            bundle_dir,
            instance_id,
            questions_by_id[question_id],
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bundle_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = json.loads(args.row_json.read_text(encoding="utf-8"))
    print(materialize(row, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
