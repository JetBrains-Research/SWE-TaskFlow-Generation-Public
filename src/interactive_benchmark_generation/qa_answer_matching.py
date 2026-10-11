from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from interactive_benchmark_generation.qa_models import (
    AnswerClaim,
    AnswerContract,
    AnswerContractKind,
    AnswerField,
    QARecord,
    normalize_answer,
)

_IDENTIFIER = r"[$A-Za-z_][$A-Za-z0-9_]*"
_REFERENCE = re.compile(rf"{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*")
_TOKEN = re.compile(rf"\s*(\?\.|\.|\[|\]|\d+|{_IDENTIFIER})")


@dataclass(frozen=True, slots=True)
class AnswerMatch:
    matched: bool
    method: str | None = None
    malformed: bool = False
    reason: str | None = None
    mismatch_path: str | None = None


def _reference_parts(value: str) -> tuple[str, ...] | None:
    value = unicodedata.normalize("NFKC", value).strip()
    if not _REFERENCE.fullmatch(value):
        return None
    return tuple(part.strip().casefold() for part in value.split("."))


def _expression_tokens(value: str) -> tuple[str, ...] | None:
    value = unicodedata.normalize("NFKC", value).strip()
    tokens = []
    position = 0
    while position < len(value):
        match = _TOKEN.match(value, position)
        if match is None:
            return None
        tokens.append(match.group(1).casefold())
        position = match.end()
    return tuple(tokens) if tokens else None


def _qualified_reference(owner: str, member: str) -> tuple[str, ...] | None:
    owner_parts = _reference_parts(owner)
    member_parts = _reference_parts(member)
    if owner_parts is None or member_parts is None or len(member_parts) != 1:
        return None
    return (*owner_parts, *member_parts)


def _join_expression(root: str, path: str) -> tuple[str, ...] | None:
    root_tokens = _expression_tokens(root)
    path_tokens = _expression_tokens(path)
    if root_tokens is None or path_tokens is None:
        return None
    separator = () if path_tokens[0] in {"[", "?."} else (".",)
    return (*root_tokens, *separator, *path_tokens)


def _legacy_contract_matches(contract: AnswerContract, answer: str) -> bool:
    if contract.kind is AnswerContractKind.EXACT:
        return False
    if contract.kind is AnswerContractKind.SYMBOL_REFERENCE:
        submitted = _reference_parts(answer)
        expected = _qualified_reference(contract.owner, contract.symbol)
        return (
            contract.allow_exact_qualification
            and submitted is not None
            and submitted == expected
        )
    if contract.kind is AnswerContractKind.MEMBER_REFERENCE:
        submitted = _reference_parts(answer)
        expected = _qualified_reference(contract.root, contract.member)
        return contract.allow_relative and submitted is not None and submitted == expected
    if contract.kind is AnswerContractKind.MEMBER_EXPRESSION:
        submitted = _expression_tokens(answer)
        full = _join_expression(contract.root, contract.path)
        return contract.allow_relative and submitted is not None and submitted == full
    parts = answer.split(";")
    if len(parts) != len(contract.components):
        return False
    for part, component in zip(parts, contract.components, strict=True):
        if part.count("=") != 1:
            return False
        role, value = part.split("=", 1)
        if normalize_answer(role) != normalize_answer(component.role):
            return False
        if normalize_answer(value) not in {
            normalize_answer(candidate) for candidate in component.accepted_values
        }:
            return False
    return True


def _malformed(reason: str, path: str | None = None) -> AnswerMatch:
    return AnswerMatch(
        matched=False,
        malformed=True,
        reason=reason,
        mismatch_path=path,
    )


def _incorrect(reason: str, path: str) -> AnswerMatch:
    return AnswerMatch(
        matched=False,
        reason=reason,
        mismatch_path=path,
    )


def _claim_shape(value: Any, path: str) -> AnswerMatch | None:
    if not isinstance(value, dict):
        return _malformed("claim_not_object", path)
    expected = {"subject", "relation", "object"}
    if set(value) != expected:
        return _malformed("invalid_claim_fields", path)
    for field in expected:
        if not isinstance(value[field], str) or not value[field].strip():
            return _malformed("claim_field_not_string", f"{path}.{field}")
    return None


def _match_field(
    expected: AnswerField,
    submitted: str,
    path: str,
    bindings: dict[str, str],
) -> AnswerMatch | None:
    normalized = normalize_answer(submitted)
    accepted = {normalize_answer(value) for value in expected.accepted_values}
    if normalized not in accepted:
        return _incorrect("field_value_not_accepted", path)
    if expected.binding is not None:
        bound = bindings.setdefault(expected.binding, normalized)
        if bound != normalized:
            return _incorrect("binding_mismatch", path)
    return None


def _match_claim(
    expected: AnswerClaim,
    submitted: Any,
    path: str,
    bindings: dict[str, str],
) -> AnswerMatch | None:
    shape_error = _claim_shape(submitted, path)
    if shape_error is not None:
        return shape_error
    for name in ("subject", "relation", "object"):
        result = _match_field(
            getattr(expected, name),
            submitted[name],
            f"{path}.{name}",
            bindings,
        )
        if result is not None:
            return result
    return None


def _match_structured_contract(
    contract: AnswerContract,
    answer: str,
) -> AnswerMatch:
    try:
        submitted = json.loads(answer)
    except json.JSONDecodeError:
        return _malformed("invalid_json")
    if not isinstance(submitted, dict):
        return _malformed("answer_not_object")
    if submitted.get("protocol") != "qa_answer/v1":
        return _malformed("wrong_protocol", "$.protocol")
    if submitted.get("kind") != contract.kind.value:
        return _malformed("wrong_kind", "$.kind")
    bindings: dict[str, str] = {}
    if contract.kind is AnswerContractKind.CAUSAL_FLOW_V1:
        expected_keys = {"protocol", "kind", "trigger", "steps", "outcome"}
        if set(submitted) != expected_keys:
            return _malformed("invalid_answer_fields", "$")
        if not isinstance(submitted["steps"], list):
            return _malformed("steps_not_array", "$.steps")
        if len(submitted["steps"]) != len(contract.steps):
            return _malformed("wrong_step_count", "$.steps")
        claims = [
            (contract.trigger, submitted["trigger"], "$.trigger"),
            *[
                (expected, actual, f"$.steps[{index}]")
                for index, (expected, actual) in enumerate(
                    zip(contract.steps, submitted["steps"], strict=True)
                )
            ],
            (contract.outcome, submitted["outcome"], "$.outcome"),
        ]
    else:
        expected_keys = {
            "protocol",
            "kind",
            "constraint",
            "decision",
            "consequence",
        }
        if set(submitted) != expected_keys:
            return _malformed("invalid_answer_fields", "$")
        claims = [
            (contract.constraint, submitted["constraint"], "$.constraint"),
            (contract.decision, submitted["decision"], "$.decision"),
            (contract.consequence, submitted["consequence"], "$.consequence"),
        ]
    for expected, actual, path in claims:
        result = _match_claim(expected, actual, path, bindings)
        if result is not None:
            return result
    return AnswerMatch(True, "answer_contract")


def match_answer(record: QARecord, answer: str) -> AnswerMatch:
    contract = record.answer_contract
    if contract is not None and contract.kind in {
        AnswerContractKind.CAUSAL_FLOW_V1,
        AnswerContractKind.RATIONALE_V1,
    }:
        return _match_structured_contract(contract, answer)
    normalized = normalize_answer(answer)
    if normalized in {normalize_answer(value) for value in record.accepted_answers}:
        return AnswerMatch(True, "exact")
    if contract is not None and _legacy_contract_matches(contract, answer):
        return AnswerMatch(True, "answer_contract")
    return AnswerMatch(False)
