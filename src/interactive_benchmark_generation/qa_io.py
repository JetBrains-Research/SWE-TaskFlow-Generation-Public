from __future__ import annotations

import errno
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from interactive_benchmark_generation.dataset_io import write_json, write_jsonl
from interactive_benchmark_generation.qa_answer_matching import match_answer
from interactive_benchmark_generation.qa_models import (
    QA_SCHEMA_V2,
    QA_SCHEMA_VERSION,
    AnswerContractKind,
    Fact,
    GenerationBundle,
    QARecord,
    normalize_answer,
)
from interactive_benchmark_generation.trajectory import load_trajectory, summarize_trajectory

_LOOKUP_QUESTION = re.compile(
    r"^\s*(?:where is|which\s+(?:property|class|method|function|enum member|identifier)|"
    r"what\s+(?:property|class|method|function|enum member|identifier|expression))\b",
    re.IGNORECASE,
)
_PROOF_REQUIRED_TOOLS = re.compile(
    r"^PROOF_REQUIRED_TOOLS=\((?P<tools>[^)]*)\)\s*$",
    re.MULTILINE,
)
_REPO_SOURCE_ASSERTION = re.compile(
    r"(?:\[\[?|\btest)\s+(?:!\s+)?-(?:e|f|r)\s+"
    r"""["']?\$(?:\{REPO_ROOT\}|REPO_ROOT)/(?P<path>[A-Za-z0-9_.+/=-]+)["']?"""
)
_BASH_ARRAY_DECLARATION = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=\((?P<values>.*?)\)\s*$",
    re.MULTILINE | re.DOTALL,
)
_BASH_ARRAY_LOOP = re.compile(
    r"""for\s+(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s+in\s+"""
    r"""["']?\$\{(?P<array>[A-Za-z_][A-Za-z0-9_]*)\[@\]\}["']?\s*;\s*"""
    r"do(?P<body>.*?)done",
    re.DOTALL,
)
# Any for-loop, over an array expansion or a literal word list.
_BASH_FOR_LOOP = re.compile(
    r"""for\s+(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s+in\s+"""
    r"""(?P<words>(?:\\\n|[^\n;])+?)\s*"""
    r"""(?:;|\n)\s*do(?P<body>.*?)done""",
    re.DOTALL,
)
_CD_TO_REPO_ROOT = re.compile(
    r"""^\s*cd\s+["']?\$(?:\{REPO_ROOT\}|REPO_ROOT)["']?\s*$""",
    re.MULTILINE,
)
_ANY_CD = re.compile(r"^\s*cd\s", re.MULTILINE)
_ARRAY_EXPANSION = re.compile(r"""^["']?\$\{(?P<array>[A-Za-z_][A-Za-z0-9_]*)\[@\]\}["']?$""")
# SRC="$REPO_ROOT/lib/x.py", optionally behind local/declare/readonly/export.
_REPO_SCALAR_ASSIGNMENT = re.compile(
    r"""^\s*(?:local|declare|readonly|export)?\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)="""
    r"""["']?(?P<value>\$(?:\{REPO_ROOT\}|REPO_ROOT)/[A-Za-z0-9_.+/=-]+)["']?\s*$""",
    re.MULTILINE,
)
# test -f "$SRC" / [[ -f "${SRC}" ]] — the variable already holds the full path.
_VARIABLE_SOURCE_ASSERTION = re.compile(
    r"""(?:\[\[?|\btest)\s+(?:!\s+)?-(?:e|f|r)\s+"""
    r"""["']?\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))["']?"""
)
_REPO_WRITE_COMMAND = re.compile(
    r"^\s*(?:sudo\s+)?(?P<command>mkdir|rm|touch)\b[^\n]*"
    r"\$(?:\{REPO_ROOT\}|REPO_ROOT)(?:/|[\"'\s]|$)",
    re.MULTILINE,
)
_DESTRUCTIVE_GIT_COMMAND = re.compile(
    r"^\s*git\s+(?P<command>checkout|clean|commit|reset|restore|switch)\b",
    re.MULTILINE,
)
_PYTHON_COMMAND = re.compile(
    r"^\s*(?:env\s+)?(?:uv\s+run\s+)?(?:python|python3|pytest)\b",
    re.MULTILINE,
)
_REPO_REDIRECTION = re.compile(
    r"(?<![<])>{1,2}\s*[\"']?\$(?:\{REPO_ROOT\}|REPO_ROOT)(?:/|[\"'])"
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"Could not read {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    return records


def load_generation_bundle(path: Path, *, prune: bool = False) -> GenerationBundle:
    """Load a generation bundle.

    `prune` drops questions that cannot be ingested instead of failing the whole
    bundle, and is for ingestion only. Re-loading an already-written bundle must
    stay strict, or corruption would be silently discarded as "unusable
    questions".
    """
    if path.is_file():
        value = _read_json(path)
        if not isinstance(value, dict):
            raise ValueError("Generation bundle must contain a JSON object")
        probes = dict(value.get("probes", {}))
        golden_proofs = dict(value.get("golden_proofs", {}))
        for question in value.get("questions", ()):
            if not isinstance(question, dict):
                continue
            probe_path = question.get("probe_path")
            if isinstance(probe_path, str) and probe_path not in probes:
                script_path = path.parent / probe_path
                if script_path.is_file():
                    probes[probe_path] = script_path.read_text()
            proof = question.get("golden_proof")
            if isinstance(proof, dict) and isinstance(proof.get("path"), str):
                proof_path = proof["path"]
                if proof_path not in golden_proofs:
                    script_path = path.parent / proof_path
                    if script_path.is_file():
                        golden_proofs[proof_path] = script_path.read_text()
        value["probes"] = probes
        value["golden_proofs"] = golden_proofs
        if prune:
            value, _ = prune_unusable_questions(value)
        return GenerationBundle.from_dict(value)
    bundle_path = path / "bundle.json"
    if bundle_path.exists():
        return load_generation_bundle(bundle_path, prune=prune)
    metadata = _read_json(path / "metadata.json")
    facts = _read_json(path / "facts.json")
    questions = _read_jsonl(path / "questions.jsonl")
    documentation = (path / "documentation.md").read_text()
    debug_path = path / "debug.json"
    debug = _read_json(debug_path) if debug_path.exists() else {}
    probes = {}
    golden_proofs = {}
    for question in questions:
        probe_path = question.get("probe_path")
        if isinstance(probe_path, str):
            probes[probe_path] = (path / probe_path).read_text()
        proof = question.get("golden_proof")
        if isinstance(proof, dict) and isinstance(proof.get("path"), str):
            golden_proofs[proof["path"]] = (path / proof["path"]).read_text()
    return GenerationBundle.from_dict(
        {
            "schema_version": metadata.get("schema_version", QA_SCHEMA_VERSION),
            "instance_id": metadata.get("instance_id"),
            "documentation": documentation,
            "facts": facts,
            "questions": questions,
            "probes": probes,
            "golden_proofs": golden_proofs,
            "metadata": metadata,
            "debug": debug,
        }
    )


def _question_rank(question: dict[str, Any]) -> tuple[int, int, int, str]:
    """Order colliding questions hardest first.

    When two questions cannot coexist, the harder one is the one worth keeping.
    Evidence hops are the direct measure: the schema demands at least three, each
    answer component needs its own hop, and at least one hop must cross files — so
    more hops means more independent components to reconstruct. Ties fall back to
    cross-file breadth, then chain length, then the question id purely so a run is
    reproducible.
    """
    hops = question.get("evidence_hops") or ()
    files = {hop.get("file") for hop in hops if isinstance(hop, dict) and hop.get("file")}
    chain = question.get("reasoning_chain") or question.get("causal_chain") or ()
    return (len(hops), len(files), len(chain), str(question.get("question_id", "")))


def _single_question_value(value: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    """The bundle as it would be with only this question, for an isolated trial."""
    proof = question.get("golden_proof")
    proof_path = proof.get("path") if isinstance(proof, dict) else None
    probe_path = question.get("probe_path")
    return {
        **value,
        "questions": [question],
        "golden_proofs": {
            path: script
            for path, script in (value.get("golden_proofs") or {}).items()
            if path == proof_path
        },
        "probes": {
            path: script
            for path, script in (value.get("probes") or {}).items()
            if path == probe_path
        },
    }


def _question_defect(value: dict[str, Any], question: dict[str, Any]) -> str | None:
    """Why this question cannot be ingested, or None if it can.

    Built by running the real constructor and validators over a bundle holding
    only this question, so the verdict cannot drift from what ingestion enforces.
    """
    try:
        bundle = GenerationBundle.from_dict(_single_question_value(value, question))
        validate_answer_contracts(bundle)
        validate_question_complexity(bundle)
        if bundle.schema_version == QA_SCHEMA_V2:
            validate_golden_proofs(bundle)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return str(error)
    return None


def prune_unusable_questions(value: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drop the questions that cannot be ingested and keep the rest.

    A defect belonging to one question — a golden proof that fails structural
    preflight, evidence ids repeated across its own hops, a missing answer
    contract — used to raise and take the whole issue with it, including sibling
    questions that were perfectly good. Now only the offending question goes.

    Two questions can also be individually valid yet unable to coexist, because a
    fact may be tested only once and question ids must be unique. There the harder
    question wins; see `_question_rank`.

    Raises only when nothing is left, which is the one case where the issue really
    has no QA to contribute.
    """
    questions = [item for item in value.get("questions", ()) if isinstance(item, dict)]
    dropped: list[dict[str, Any]] = []

    kept: list[dict[str, Any]] = []
    for question in questions:
        defect = _question_defect(value, question)
        if defect is None:
            kept.append(question)
        else:
            dropped.append({
                "question_id": question.get("question_id"),
                "reason": "question_defect",
                "detail": defect,
            })

    # Individually valid but mutually exclusive: keep the hardest of each group.
    for key, reason in (("fact_id", "duplicate_fact"), ("question_id", "duplicate_question_id")):
        groups: dict[Any, list[dict[str, Any]]] = {}
        for question in kept:
            groups.setdefault(question.get(key), []).append(question)
        survivors = []
        for group in groups.values():
            winner, *losers = sorted(group, key=_question_rank, reverse=True)
            survivors.append(winner)
            for loser in losers:
                dropped.append({
                    "question_id": loser.get("question_id"),
                    "reason": reason,
                    "detail": (
                        f"collides with {winner.get('question_id')} on {key}; "
                        f"kept the harder question (rank {_question_rank(winner)} "
                        f"beats {_question_rank(loser)})"
                    ),
                })
        kept = [question for question in kept if question in survivors]

    if not kept:
        details = "; ".join(f"{item['question_id']}: {item['detail']}" for item in dropped)
        raise ValueError(f"No ingestible questions remain: {details}")

    pruned = dict(value)
    pruned["questions"] = kept
    proof_paths = {
        question["golden_proof"]["path"]
        for question in kept
        if isinstance(question.get("golden_proof"), dict)
    }
    probe_paths = {question.get("probe_path") for question in kept}
    pruned["golden_proofs"] = {
        path: script
        for path, script in (value.get("golden_proofs") or {}).items()
        if path in proof_paths
    }
    pruned["probes"] = {
        path: script
        for path, script in (value.get("probes") or {}).items()
        if path in probe_paths
    }
    if dropped:
        pruned["debug"] = {
            **(value.get("debug") or {}),
            "ingestion_dropped_questions": dropped,
        }
    return pruned, dropped


def load_qa_records(bundle_dir: Path) -> list[QARecord]:
    return [QARecord.from_dict(value) for value in _read_jsonl(bundle_dir / "questions.jsonl")]


def _repo_source_path(value: str) -> tuple[str, bool] | None:
    """Normalise a literal repository path.

    Returns the repository-relative path and whether it was written with an
    explicit ``$REPO_ROOT/`` prefix, or None when the value is not a concrete
    literal path inside the repository.
    """
    rooted = False
    for prefix in ("${REPO_ROOT}/", "$REPO_ROOT/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            rooted = True
            break
    if not value or "$" in value:
        return None
    if re.fullmatch(r"[A-Za-z0-9_.+/=-]+", value) is None:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    if value == ".git" or value.startswith(".git/"):
        return None
    return value, rooted


def _split_literal_words(words: str) -> list[tuple[str, bool]] | None:
    # Bash removes a backslash-newline outright; shlex would keep it as a token.
    try:
        raw = shlex.split(words.replace("\\\n", " "))
    except ValueError:
        return None
    if not raw:
        return None
    resolved = [_repo_source_path(value) for value in raw]
    if any(item is None for item in resolved):
        return None
    return resolved  # type: ignore[return-value]


def _loop_body_asserts(body: str, variable: str, *, rooted_values: bool) -> bool:
    """Does the loop body test the loop variable as a file under REPO_ROOT?

    When the values already carry ``$REPO_ROOT/`` the body may test the bare
    variable; otherwise it must join the value onto REPO_ROOT itself. Anything
    weaker would stop proving that the asserted file is the repository's.
    """
    name = re.escape(variable)
    if rooted_values:
        pattern = (
            r"""(?:\[\[?|\btest)\s+(?:!\s+)?-(?:e|f|r)\s+"""
            r"""["']?\$(?:\{""" + name + r"""\}|""" + name + r""")["']?"""
        )
    else:
        pattern = (
            r"""(?:\[\[?|\btest)\s+(?:!\s+)?-(?:e|f|r)\s+["']?"""
            r"""\$(?:\{REPO_ROOT\}|REPO_ROOT)/\$(?:\{""" + name + r"""\}|""" + name + r""")["']?"""
        )
    return re.search(pattern, body) is not None


def _cwd_is_repo_root(script: str, position: int) -> bool:
    """Has the script cd'd to REPO_ROOT and stayed there, as of `position`?

    Only then does testing a bare relative path prove a file under REPO_ROOT.
    """
    prefix = script[:position]
    anchored = [match.end() for match in _CD_TO_REPO_ROOT.finditer(prefix)]
    if not anchored:
        return False
    return not any(match.start() >= anchored[-1] for match in _ANY_CD.finditer(prefix))


def _collect_source_assertions(script: str) -> tuple[list[str], list[int]]:
    """Every repository source file the script proves exists, and where it does so.

    Four idioms are accepted, because a proof that asserts its source files is
    correct regardless of which one it picks:

    1. ``test -f "$REPO_ROOT/lib/x.py"`` — the path inline.
    2. ``SRC="$REPO_ROOT/lib/x.py"`` … ``test -f "$SRC"`` — through a scalar.
    3. ``SRCS=(lib/x.py lib/y.py)`` … ``for s in "${SRCS[@]}"; do test -f "$REPO_ROOT/$s"``.
    4. ``for s in lib/x.py lib/y.py; do test -f "$REPO_ROOT/$s"`` — a literal word list.

    Only 1 and 3 used to be recognised, so a proof using 2 or 4 was rejected as
    if it had asserted nothing.
    """
    paths: list[str] = []
    positions: list[int] = []

    for match in _REPO_SOURCE_ASSERTION.finditer(script):
        resolved = _repo_source_path("$REPO_ROOT/" + match.group("path"))
        if resolved is not None:
            paths.append(resolved[0])
            positions.append(match.start())

    scalars = {}
    for match in _REPO_SCALAR_ASSIGNMENT.finditer(script):
        resolved = _repo_source_path(match.group("value"))
        if resolved is not None:
            scalars[match.group("name")] = resolved[0]
    for match in _VARIABLE_SOURCE_ASSERTION.finditer(script):
        value = scalars.get(match.group("braced") or match.group("bare"))
        if value is not None:
            paths.append(value)
            positions.append(match.start())

    arrays = {}
    for match in _BASH_ARRAY_DECLARATION.finditer(script):
        resolved = _split_literal_words(match.group("values"))
        if resolved is not None:
            arrays[match.group("name")] = resolved

    for match in _BASH_FOR_LOOP.finditer(script):
        expansion = _ARRAY_EXPANSION.match(match.group("words").strip())
        resolved = arrays.get(expansion.group("array")) if expansion else _split_literal_words(match.group("words"))
        if not resolved:
            continue
        # A mixed list would make the required assertion form ambiguous.
        if len({rooted for _, rooted in resolved}) != 1:
            continue
        forms = {resolved[0][1]}
        # Bare relative values tested bare are still under REPO_ROOT when the
        # script has cd'd there and has not cd'd anywhere else since.
        if not resolved[0][1] and _cwd_is_repo_root(script, match.start()):
            forms.add(True)
        if any(
            _loop_body_asserts(match.group("body"), match.group("variable"), rooted_values=form)
            for form in forms
        ):
            paths.extend(value for value, _ in resolved)
            positions.append(match.start())

    return list(dict.fromkeys(paths)), positions


def _validate_golden_proof_source(script: str, proof_path: str) -> dict[str, Any]:
    errors = []
    if "set -euo pipefail" not in script:
        errors.append("must enable set -euo pipefail")
    if "REPO_ROOT" not in script:
        errors.append("must use REPO_ROOT")
    source_paths, source_assertion_positions = _collect_source_assertions(script)
    if not source_paths:
        errors.append(
            "must explicitly assert at least one exact source file under REPO_ROOT"
        )
    tools_match = _PROOF_REQUIRED_TOOLS.search(script)
    required_tools = []
    if tools_match is None:
        errors.append("must declare PROOF_REQUIRED_TOOLS=(...)")
    else:
        try:
            required_tools = shlex.split(tools_match.group("tools"))
        except ValueError as error:
            errors.append(f"has an invalid PROOF_REQUIRED_TOOLS declaration: {error}")
        if not required_tools:
            errors.append("must identify at least one required proof tool")
        invalid_tools = [
            tool
            for tool in required_tools
            if re.fullmatch(r"[A-Za-z0-9_.+-]+", tool) is None
        ]
        if invalid_tools:
            errors.append(f"has invalid required proof tools: {invalid_tools}")
    if "command -v" not in script or "missing required proof tool" not in script:
        errors.append(
            "must check required tools with command -v and an explicit "
            "'missing required proof tool' diagnostic"
        )
    python_commands = list(_PYTHON_COMMAND.finditer(script))
    python_used = any(
        tool in {"python", "python3", "pytest"} for tool in required_tools
    ) or bool(python_commands)
    if python_used:
        for setting in (
            "PYTHONPATH",
            "PYTHONNOUSERSITE=1",
            "PYTHONSAFEPATH=1",
            "PYTHONDONTWRITEBYTECODE=1",
        ):
            if setting not in script:
                errors.append(f"Python proof must configure {setting}")
        if not re.search(
            r"""PYTHONPATH=["']?\$(?:\{REPO_ROOT\}|REPO_ROOT)(?::[^"'\s]+)?["']?""",
            script,
        ):
            errors.append("Python proof must anchor PYTHONPATH at REPO_ROOT")
        if not re.search(
            r"""^\s*cd\s+["']?\$(?:\{REPO_ROOT\}|REPO_ROOT)["']?\s*$""",
            script,
            re.MULTILINE,
        ):
            errors.append("Python proof must change directory to REPO_ROOT")
        if python_commands and (
            not source_paths
            or min(source_assertion_positions) > python_commands[0].start()
        ):
            errors.append(
                "Python proof must assert repository source before running Python"
            )
    write_commands = sorted(
        {
            match.group("command")
            for pattern in (_REPO_WRITE_COMMAND, _DESTRUCTIVE_GIT_COMMAND)
            for match in pattern.finditer(script)
        }
    )
    if write_commands:
        errors.append(
            "contains commands that may write repository state: "
            f"{', '.join(write_commands)}"
        )
    if _REPO_REDIRECTION.search(script):
        errors.append("contains output redirection into REPO_ROOT")
    if errors:
        raise ValueError(
            f"Golden proof structural preflight failed for {proof_path}: "
            + "; ".join(errors)
        )
    return {
        "required_tools": required_tools,
        "asserted_source_files": source_paths,
        "python_bytecode_disabled": not python_used
        or "PYTHONDONTWRITEBYTECODE=1" in script,
        "repository_write_commands": write_commands,
    }


def validate_golden_proofs(
    bundle: GenerationBundle,
    bundle_dir: Path | None = None,
    *,
    require_executable: bool = True,
) -> dict[str, Any]:
    if bundle.schema_version != QA_SCHEMA_V2:
        return {
            "schema_version": bundle.schema_version,
            "n_questions": len(bundle.questions),
            "n_valid": 0,
            "valid": True,
            "proofs": [],
        }
    paths = [record.golden_proof.path for record in bundle.questions]
    if len(set(paths)) != len(paths):
        raise ValueError("Golden proof paths must be unique")
    if set(paths) != set(bundle.golden_proofs):
        raise ValueError("Golden proof paths must map one-to-one to questions")
    results = []
    root = bundle_dir.resolve() if bundle_dir is not None else None
    for record in bundle.questions:
        proof = record.golden_proof
        script = bundle.golden_proofs[proof.path]
        if not isinstance(script, str) or not script.strip():
            raise ValueError(f"Golden proof is empty: {proof.path}")
        first_line = script.splitlines()[0]
        if first_line not in {"#!/usr/bin/env bash", "#!/bin/bash"}:
            raise ValueError(f"Golden proof must use a Bash shebang: {proof.path}")
        source_preflight = _validate_golden_proof_source(script, proof.path)
        digest = hashlib.sha256(script.encode()).hexdigest()
        if digest != proof.sha256:
            raise ValueError(f"Golden proof SHA-256 mismatch: {proof.path}")
        executable = True
        if root is not None:
            target = (root / proof.path).resolve()
            if root not in target.parents or not target.is_file():
                raise ValueError(f"Missing golden proof file: {proof.path}")
            executable = bool(target.stat().st_mode & 0o111)
            if require_executable and not executable:
                raise ValueError(f"Golden proof is not executable: {proof.path}")
        results.append(
            {
                "question_id": record.question_id,
                "path": proof.path,
                "sha256": digest,
                "executable": executable,
                **source_preflight,
            }
        )
    return {
        "schema_version": bundle.schema_version,
        "n_questions": len(bundle.questions),
        "n_valid": len(results),
        "valid": True,
        "proofs": results,
    }


def validate_answer_contracts(bundle: GenerationBundle) -> None:
    for record in bundle.questions:
        if record.schema_version == QA_SCHEMA_V2:
            continue
        contract = record.answer_contract
        if contract is None:
            raise ValueError(f"Missing answer_contract for {record.question_id}")
        canonical = None
        if contract.kind is AnswerContractKind.SYMBOL_REFERENCE:
            canonical = contract.symbol
        elif contract.kind is AnswerContractKind.MEMBER_REFERENCE:
            canonical = contract.member
        elif contract.kind is AnswerContractKind.MEMBER_EXPRESSION:
            canonical = contract.path
        elif contract.kind is AnswerContractKind.CAUSAL_CHAIN:
            canonical = "; ".join(
                f"{component.role}={component.accepted_values[0]}"
                for component in contract.components
            )
        elif contract.kind in {
            AnswerContractKind.CAUSAL_FLOW_V1,
            AnswerContractKind.RATIONALE_V1,
        }:
            canonical = contract.canonical_answer()
        if canonical is not None and normalize_answer(canonical) not in {
            normalize_answer(answer) for answer in record.accepted_answers
        }:
            raise ValueError(
                f"Canonical contract answer is not accepted for {record.question_id}: "
                f"{canonical}"
            )
        if contract.kind in {
            AnswerContractKind.CAUSAL_FLOW_V1,
            AnswerContractKind.RATIONALE_V1,
        }:
            if len(record.accepted_answers) != 1:
                raise ValueError(
                    f"Structured contracts require one canonical accepted answer: "
                    f"{record.question_id}"
                )
            malformed_incorrect = [
                answer
                for answer in record.incorrect_answers
                if match_answer(record, answer).malformed
            ]
            if malformed_incorrect:
                raise ValueError(
                    f"Structured incorrect answers must use valid protocol JSON for "
                    f"{record.question_id}: {malformed_incorrect}"
                )
        structurally_accepted_incorrect = [
            answer for answer in record.incorrect_answers if match_answer(record, answer).matched
        ]
        if structurally_accepted_incorrect:
            raise ValueError(
                f"answer_contract accepts incorrect answers for {record.question_id}: "
                f"{structurally_accepted_incorrect}"
            )


def validate_question_complexity(bundle: GenerationBundle) -> None:
    for record in bundle.questions:
        if record.category is None:
            raise ValueError(f"Missing category for {record.question_id}")
        if len(record.evidence_hops) < 3:
            raise ValueError(f"Insufficient evidence hops for {record.question_id}")
        if len(record.causal_chain) < 3:
            raise ValueError(f"Insufficient causal chain for {record.question_id}")
        if _LOOKUP_QUESTION.match(record.question):
            raise ValueError(f"Lookup question is not allowed: {record.question_id}")


def load_facts(bundle_dir: Path) -> list[Fact]:
    value = _read_json(bundle_dir / "facts.json")
    if not isinstance(value, list):
        raise ValueError("facts.json must contain a JSON array")
    return [Fact.from_dict(item) for item in value]


def validate_record_relationships(facts: list[Fact], records: list[QARecord]) -> None:
    fact_ids = [fact.fact_id for fact in facts]
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError("Fact identifiers must be unique")
    question_ids = [record.question_id for record in records]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Question identifiers must be unique")
    tested_facts = [record.fact_id for record in records]
    if len(set(tested_facts)) != len(tested_facts):
        raise ValueError("Each fact may be tested by at most one question")
    facts_by_id = {fact.fact_id: fact for fact in facts}
    for record in records:
        fact = facts_by_id.get(record.fact_id)
        if fact is None:
            raise ValueError(f"Unknown fact_id for {record.question_id}: {record.fact_id}")
        if record.fact != fact.fact:
            raise ValueError(f"Fact text mismatch for {record.question_id}")
        if record.instance_id != fact.instance_id:
            raise ValueError(f"Instance mismatch for {record.question_id}")


def write_qa_bundle(
    bundle: GenerationBundle,
    output_dir: Path,
    trajectory_path: Path | None = None,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_summary = None
    if trajectory_path is not None:
        trajectory_summary = summarize_trajectory(load_trajectory(trajectory_path))
    (output_dir / "documentation.md").write_text(bundle.documentation.rstrip() + "\n")
    write_json(output_dir / "facts.json", [fact.to_dict() for fact in bundle.facts])
    write_jsonl(output_dir / "questions.jsonl", [record.to_dict() for record in bundle.questions])
    metadata = {
        **bundle.metadata,
        "schema_version": bundle.schema_version,
        "instance_id": bundle.instance_id,
        "n_facts": len(bundle.facts),
        "n_questions": len(bundle.questions),
    }
    if trajectory_summary is not None:
        metadata["trajectory"] = trajectory_summary
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "debug.json", bundle.debug)
    for relative_path, script in bundle.probes.items():
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(script.rstrip() + "\n")
        try:
            target.chmod(0o755)
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
    for relative_path, script in bundle.golden_proofs.items():
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(script)
        try:
            target.chmod(0o755)
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
    validate_golden_proofs(bundle, output_dir, require_executable=False)


def replace_questions(bundle_dir: Path, records: list[QARecord]) -> None:
    target_path = bundle_dir / "questions.jsonl"
    temporary_path = bundle_dir / f".questions.jsonl.{uuid4().hex}.tmp"
    try:
        write_jsonl(temporary_path, [record.to_dict() for record in records])
        content = temporary_path.read_text(encoding="utf-8")
        try:
            temporary_path.replace(target_path)
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
            target_path.write_text(content, encoding="utf-8")
            if target_path.read_text(encoding="utf-8") != content:
                raise OSError(
                    f"Failed to persist questions: {target_path}"
                ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def copy_qa_bundle(source_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    shutil.copytree(source_dir, output_dir)
