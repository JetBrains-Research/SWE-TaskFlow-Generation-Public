from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

QA_SCHEMA_VERSION = "1.0"
QA_SCHEMA_V2 = "2.0"
QA_PROOF_JUDGE_PROTOCOL = "qa_proof_judge/v1"
QA_ANSWER_JUDGE_PROTOCOL = "qa_answer_judge/v1"
QA_PROOF_EXECUTION_PROTOCOL = "qa_proof_execution/v1"
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Answers must be strings")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _documentation_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)) and value:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    raise ValueError("documentation must be a non-empty string or structured value")


def _validate_schema_version(value: object) -> str:
    if value in {1, "1", "qa_benchmark_bundle/v1"}:
        value = QA_SCHEMA_VERSION
    if value in {2, "2", "qa_benchmark_bundle/v2"}:
        value = QA_SCHEMA_V2
    if value not in {QA_SCHEMA_VERSION, QA_SCHEMA_V2}:
        raise ValueError(f"Unsupported QA schema version: {value!r}")
    return value


def _safe_relative_path(value: object, name: str, prefix: str | None = None) -> str:
    path_value = _required_string(value, name)
    path = PurePosixPath(path_value)
    if path.is_absolute() or ".." in path.parts or str(path) != path_value:
        raise ValueError(f"{name} must be a safe relative path")
    if prefix is not None and (not path.parts or path.parts[0] != prefix):
        raise ValueError(f"{name} must be under {prefix}/")
    return path_value


def _sha256(value: object, name: str = "sha256") -> str:
    digest = _required_string(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _strict_keys(value: dict[str, Any], required: set[str], optional: set[str], name: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class Fact:
    instance_id: str
    fact_id: str
    fact: str
    schema_version: str = QA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _validate_schema_version(self.schema_version),
        )
        _required_string(self.instance_id, "instance_id")
        _required_string(self.fact_id, "fact_id")
        _required_string(self.fact, "fact")

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        default_instance_id: str | None = None,
        default_schema_version: str = QA_SCHEMA_VERSION,
    ) -> Fact:
        return cls(
            instance_id=value.get("instance_id", default_instance_id),
            fact_id=value.get("fact_id"),
            fact=value.get("fact"),
            schema_version=value.get("schema_version", default_schema_version),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeValidation:
    accepted_passed: bool
    incorrect_rejected: bool
    runs: tuple[dict[str, Any], ...] = ()
    schema_version: str = QA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _validate_schema_version(self.schema_version),
        )
        if not isinstance(self.accepted_passed, bool):
            raise TypeError("accepted_passed must be a boolean")
        if not isinstance(self.incorrect_rejected, bool):
            raise TypeError("incorrect_rejected must be a boolean")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProbeValidation:
        runs = value.get("runs", ())
        if not isinstance(runs, (list, tuple)) or not all(isinstance(run, dict) for run in runs):
            raise TypeError("runs must be a list of objects")
        return cls(
            accepted_passed=value.get("accepted_passed"),
            incorrect_rejected=value.get("incorrect_rejected"),
            runs=tuple(runs),
            schema_version=value.get("schema_version", QA_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runs"] = list(self.runs)
        return value


@dataclass(frozen=True, slots=True)
class AdversarialReview:
    accepted: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...] = ()
    rejection_codes: tuple[str, ...] = ()
    defeating_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean")
        if not self.checks or any(
            not isinstance(name, str) or not isinstance(result, bool)
            for name, result in self.checks.items()
        ):
            raise TypeError("checks must be a non-empty boolean mapping")
        if self.accepted != all(self.checks.values()):
            raise ValueError("accepted must equal the conjunction of review checks")
        for reason in self.reasons:
            _required_string(reason, "reason")
        for code in self.rejection_codes:
            _required_string(code, "rejection_code")
        for path in self.defeating_paths:
            _required_string(path, "defeating_path")
        if self.accepted and (self.rejection_codes or self.defeating_paths):
            raise ValueError("Accepted reviews must not contain rejection diagnostics")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AdversarialReview:
        return cls(
            accepted=value.get("accepted"),
            checks=dict(value.get("checks", {})),
            reasons=tuple(value.get("reasons", ())),
            rejection_codes=tuple(value.get("rejection_codes", ())),
            defeating_paths=tuple(value.get("defeating_paths", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "rejection_codes": list(self.rejection_codes),
            "defeating_paths": list(self.defeating_paths),
        }


class AnswerContractKind(StrEnum):
    EXACT = "exact"
    SYMBOL_REFERENCE = "symbol_reference"
    MEMBER_REFERENCE = "member_reference"
    MEMBER_EXPRESSION = "member_expression"
    CAUSAL_CHAIN = "causal_chain"
    CAUSAL_FLOW_V1 = "causal_flow_v1"
    RATIONALE_V1 = "rationale_v1"


class CausalFlowRelation(StrEnum):
    TRIGGERS = "triggers"
    ENABLES = "enables"
    GUARDS = "guards"
    SELECTS = "selects"
    TRANSFORMS = "transforms"
    PROPAGATES = "propagates"
    SUPPRESSES = "suppresses"
    CAUSES = "causes"
    RETURNS = "returns"
    READS = "reads"
    WRITES = "writes"
    MAPS = "maps"
    QUERIES = "queries"
    CALLS = "calls"


class RationaleRelation(StrEnum):
    PREMISE = "premise"
    CONSTRAINT = "constraint"
    MECHANISM = "mechanism"
    TRADEOFF = "tradeoff"
    JUSTIFIES = "justifies"
    THEREFORE = "therefore"


class QuestionCategory(StrEnum):
    WHAT_ARCHITECTURE_EXPLORATION = "what_architecture_exploration"
    WHAT_CONCEPT_DEFINITION = "what_concept_definition"
    WHAT_DEPENDENCY_TRACING = "what_dependency_tracing"
    WHY_DESIGN_RATIONALE = "why_design_rationale"
    WHY_PURPOSE_EXPLORATION = "why_purpose_exploration"
    WHY_PERFORMANCE = "why_performance"
    WHERE_DATA_CONTROL_FLOW = "where_data_control_flow"
    HOW_SYSTEM_DESIGN = "how_system_design"
    HOW_ALGORITHM_IMPLEMENTATION = "how_algorithm_implementation"
    HOW_API_FRAMEWORK_SUPPORT = "how_api_framework_support"


@dataclass(frozen=True, slots=True)
class EvidenceHop:
    evidence_id: str
    file: str
    symbol: str
    claim: str

    def __post_init__(self) -> None:
        for name in ("evidence_id", "file", "symbol", "claim"):
            _required_string(getattr(self, name), name)
        path = PurePosixPath(self.file)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Evidence file must be a safe repository-relative path")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceHop:
        return cls(
            evidence_id=value.get("evidence_id"),
            file=value.get("file"),
            symbol=value.get("symbol"),
            claim=value.get("claim"),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FlowScope:
    entry_symbol: str
    terminal_symbol: str

    def __post_init__(self) -> None:
        _required_string(self.entry_symbol, "entry_symbol")
        _required_string(self.terminal_symbol, "terminal_symbol")
        if normalize_answer(self.entry_symbol) == normalize_answer(self.terminal_symbol):
            raise ValueError("Flow scope entry and terminal symbols must differ")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FlowScope:
        return cls(
            entry_symbol=value.get("entry_symbol"),
            terminal_symbol=value.get("terminal_symbol"),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CausalStep:
    step_id: str
    claim: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _required_string(self.step_id, "step_id")
        _required_string(self.claim, "claim")
        if not self.evidence_ids:
            raise ValueError("Causal steps must reference evidence")
        for evidence_id in self.evidence_ids:
            _required_string(evidence_id, "evidence_id")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CausalStep:
        return cls(
            step_id=value.get("step_id"),
            claim=value.get("claim"),
            evidence_ids=tuple(value.get("evidence_ids", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_ids"] = list(self.evidence_ids)
        return value


@dataclass(frozen=True, slots=True)
class CausalAnswerComponent:
    role: str
    accepted_values: tuple[str, ...]

    def __post_init__(self) -> None:
        _required_string(self.role, "role")
        if not self.accepted_values:
            raise ValueError("Causal answer components require accepted values")
        for value in self.accepted_values:
            _required_string(value, "accepted_value")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CausalAnswerComponent:
        return cls(
            role=value.get("role"),
            accepted_values=tuple(value.get("accepted_values", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "accepted_values": list(self.accepted_values)}


@dataclass(frozen=True, slots=True)
class AnswerField:
    accepted_values: tuple[str, ...]
    binding: str | None = None

    def __post_init__(self) -> None:
        if not self.accepted_values:
            raise ValueError("Structured answer fields require accepted values")
        for value in self.accepted_values:
            _required_string(value, "accepted_value")
        normalized = [normalize_answer(value) for value in self.accepted_values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Structured field accepted values must be unique")
        if self.binding is not None:
            _required_string(self.binding, "binding")

    @classmethod
    def from_dict(cls, value: object) -> AnswerField:
        if isinstance(value, str):
            return cls(accepted_values=(value,))
        if not isinstance(value, dict):
            raise TypeError("Structured answer fields must be strings or objects")
        unknown = set(value) - {"accepted_values", "binding"}
        if unknown:
            raise ValueError(f"Unknown structured answer field keys: {sorted(unknown)}")
        return cls(
            accepted_values=tuple(value.get("accepted_values", ())),
            binding=value.get("binding"),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"accepted_values": list(self.accepted_values)}
        if self.binding is not None:
            value["binding"] = self.binding
        return value


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    subject: AnswerField
    relation: AnswerField
    object: AnswerField

    @classmethod
    def from_dict(cls, value: object) -> AnswerClaim:
        if not isinstance(value, dict):
            raise TypeError("Structured answer claims must be objects")
        expected = {"subject", "relation", "object"}
        if set(value) != expected:
            raise ValueError(
                f"Structured answer claims require exactly: {sorted(expected)}"
            )
        return cls(
            subject=AnswerField.from_dict(value["subject"]),
            relation=AnswerField.from_dict(value["relation"]),
            object=AnswerField.from_dict(value["object"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject.to_dict(),
            "relation": self.relation.to_dict(),
            "object": self.object.to_dict(),
        }

    def canonical_value(self) -> dict[str, str]:
        return {
            "subject": self.subject.accepted_values[0],
            "relation": self.relation.accepted_values[0],
            "object": self.object.accepted_values[0],
        }


@dataclass(frozen=True, slots=True)
class AnswerContract:
    kind: AnswerContractKind | str
    protocol: AnswerField | None = None
    answer_kind: AnswerField | None = None
    symbol: str | None = None
    owner: str | None = None
    root: str | None = None
    member: str | None = None
    path: str | None = None
    components: tuple[CausalAnswerComponent, ...] = ()
    trigger: AnswerClaim | None = None
    steps: tuple[AnswerClaim, ...] = ()
    outcome: AnswerClaim | None = None
    constraint: AnswerClaim | None = None
    decision: AnswerClaim | None = None
    consequence: AnswerClaim | None = None
    allow_exact_qualification: bool = False
    allow_relative: bool = False
    schema_version: str = QA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_schema_version(self.schema_version))
        object.__setattr__(self, "kind", AnswerContractKind(self.kind))
        if not isinstance(self.allow_exact_qualification, bool):
            raise TypeError("allow_exact_qualification must be a boolean")
        if not isinstance(self.allow_relative, bool):
            raise TypeError("allow_relative must be a boolean")
        if self.kind is AnswerContractKind.EXACT:
            self._reject_structured_fields()
            self._reject_fields(
                "protocol",
                "answer_kind",
                "symbol",
                "owner",
                "root",
                "member",
                "path",
                "components",
            )
        elif self.kind is AnswerContractKind.SYMBOL_REFERENCE:
            self._reject_structured_fields()
            self._reject_fields("protocol", "answer_kind")
            _required_string(self.symbol, "symbol")
            _required_string(self.owner, "owner")
            self._reject_fields("root", "member", "path", "components")
        elif self.kind is AnswerContractKind.MEMBER_REFERENCE:
            self._reject_structured_fields()
            self._reject_fields("protocol", "answer_kind")
            _required_string(self.root, "root")
            _required_string(self.member, "member")
            self._reject_fields("symbol", "owner", "path", "components")
        elif self.kind is AnswerContractKind.MEMBER_EXPRESSION:
            self._reject_structured_fields()
            self._reject_fields("protocol", "answer_kind")
            _required_string(self.root, "root")
            _required_string(self.path, "path")
            self._reject_fields("symbol", "owner", "member", "components")
        elif self.kind is AnswerContractKind.CAUSAL_CHAIN:
            self._reject_structured_fields()
            self._reject_fields("protocol", "answer_kind")
            self._reject_fields("symbol", "owner", "root", "member", "path")
            if len(self.components) < 3:
                raise ValueError("causal_chain requires at least three components")
            roles = [normalize_answer(component.role) for component in self.components]
            if len(set(roles)) != len(roles):
                raise ValueError("Causal answer component roles must be unique")
        elif self.kind is AnswerContractKind.CAUSAL_FLOW_V1:
            self._set_and_validate_protocol_fields()
            self._reject_fields("symbol", "owner", "root", "member", "path", "components")
            if self.trigger is None or self.outcome is None or len(self.steps) < 1:
                raise ValueError("causal_flow_v1 requires trigger, steps, and outcome")
            self._reject_fields("constraint", "decision", "consequence")
            self._validate_relations(CausalFlowRelation)
        else:
            self._set_and_validate_protocol_fields()
            self._reject_fields("symbol", "owner", "root", "member", "path", "components")
            if self.constraint is None or self.decision is None or self.consequence is None:
                raise ValueError(
                    "rationale_v1 requires constraint, decision, and consequence"
                )
            self._reject_fields("trigger", "steps", "outcome")
            self._validate_relations(RationaleRelation)

    def _set_and_validate_protocol_fields(self) -> None:
        if self.protocol is None:
            object.__setattr__(
                self,
                "protocol",
                AnswerField(accepted_values=("qa_answer/v1",)),
            )
        if self.answer_kind is None:
            object.__setattr__(
                self,
                "answer_kind",
                AnswerField(accepted_values=(self.kind.value,)),
            )
        if tuple(map(normalize_answer, self.protocol.accepted_values)) != (
            "qa_answer/v1",
        ):
            raise ValueError("Structured protocol must be exactly qa_answer/v1")
        if tuple(map(normalize_answer, self.answer_kind.accepted_values)) != (
            self.kind.value,
        ):
            raise ValueError(
                f"Structured kind field must be exactly {self.kind.value}"
            )

    def _reject_structured_fields(self) -> None:
        self._reject_fields(
            "trigger",
            "steps",
            "outcome",
            "constraint",
            "decision",
            "consequence",
        )

    def _structured_claims(self) -> tuple[AnswerClaim, ...]:
        if self.kind is AnswerContractKind.CAUSAL_FLOW_V1:
            return (self.trigger, *self.steps, self.outcome)
        if self.kind is AnswerContractKind.RATIONALE_V1:
            return (self.constraint, self.decision, self.consequence)
        return ()

    def _validate_relations(self, relation_kind: type[StrEnum]) -> None:
        allowed = {item.value for item in relation_kind}
        for claim in self._structured_claims():
            for relation in claim.relation.accepted_values:
                if normalize_answer(relation) not in allowed:
                    raise ValueError(
                        f"Unsupported {self.kind.value} relation: {relation}"
                    )

    def _reject_fields(self, *names: str) -> None:
        populated = [
            name
            for name in names
            if getattr(self, name) not in {None, ()}
        ]
        if populated:
            raise ValueError(
                f"{self.kind.value} does not accept fields: {', '.join(populated)}"
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AnswerContract:
        if "schema" in value:
            schema = value["schema"]
            if not isinstance(schema, dict):
                raise TypeError("answer_contract.schema must be an object")
            unknown_outer = set(value) - {"type", "kind", "schema", "schema_version"}
            if unknown_outer:
                raise ValueError(
                    f"Unknown answer_contract wrapper keys: {sorted(unknown_outer)}"
                )
            flattened = {
                "type": value.get("type", value.get("kind")),
                **schema,
            }
            if "schema_version" in value:
                flattened["schema_version"] = value["schema_version"]
            return cls.from_dict(flattened)
        kind = AnswerContractKind(value.get("type", value.get("kind")))
        structured_keys = {
            "trigger",
            "steps",
            "outcome",
            "constraint",
            "decision",
            "consequence",
        }
        common_keys = {
            "type",
            "kind",
            "symbol",
            "owner",
            "root",
            "member",
            "path",
            "components",
            "allow_exact_qualification",
            "allow_relative",
            "schema_version",
            "protocol",
            *structured_keys,
        }
        unknown = set(value) - common_keys
        if unknown:
            raise ValueError(f"Unknown answer_contract keys: {sorted(unknown)}")
        return cls(
            kind=kind,
            protocol=(
                AnswerField.from_dict(value["protocol"])
                if value.get("protocol") is not None
                else None
            ),
            answer_kind=(
                AnswerField.from_dict(value["kind"])
                if isinstance(value.get("kind"), dict)
                else None
            ),
            symbol=value.get("symbol"),
            owner=value.get("owner"),
            root=value.get("root"),
            member=value.get("member"),
            path=value.get("path"),
            components=tuple(
                CausalAnswerComponent.from_dict(item)
                for item in value.get("components", ())
            ),
            trigger=(
                AnswerClaim.from_dict(value["trigger"])
                if value.get("trigger") is not None
                else None
            ),
            steps=tuple(AnswerClaim.from_dict(item) for item in value.get("steps", ())),
            outcome=(
                AnswerClaim.from_dict(value["outcome"])
                if value.get("outcome") is not None
                else None
            ),
            constraint=(
                AnswerClaim.from_dict(value["constraint"])
                if value.get("constraint") is not None
                else None
            ),
            decision=(
                AnswerClaim.from_dict(value["decision"])
                if value.get("decision") is not None
                else None
            ),
            consequence=(
                AnswerClaim.from_dict(value["consequence"])
                if value.get("consequence") is not None
                else None
            ),
            allow_exact_qualification=value.get("allow_exact_qualification", False),
            allow_relative=value.get("allow_relative", False),
            schema_version=value.get("schema_version", QA_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": self.kind.value,
            "allow_exact_qualification": self.allow_exact_qualification,
            "allow_relative": self.allow_relative,
            "schema_version": self.schema_version,
        }
        if self.protocol is not None:
            value["protocol"] = self.protocol.to_dict()
        if self.answer_kind is not None:
            value["kind"] = self.answer_kind.to_dict()
        for name in ("symbol", "owner", "root", "member", "path"):
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        if self.components:
            value["components"] = [component.to_dict() for component in self.components]
        for name in ("trigger", "outcome", "constraint", "decision", "consequence"):
            claim = getattr(self, name)
            if claim is not None:
                value[name] = claim.to_dict()
        if self.steps:
            value["steps"] = [claim.to_dict() for claim in self.steps]
        return value

    def canonical_answer(self) -> str | None:
        if self.kind is AnswerContractKind.CAUSAL_FLOW_V1:
            value = {
                "protocol": "qa_answer/v1",
                "kind": self.kind.value,
                "trigger": self.trigger.canonical_value(),
                "steps": [claim.canonical_value() for claim in self.steps],
                "outcome": self.outcome.canonical_value(),
            }
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if self.kind is AnswerContractKind.RATIONALE_V1:
            value = {
                "protocol": "qa_answer/v1",
                "kind": self.kind.value,
                "constraint": self.constraint.canonical_value(),
                "decision": self.decision.canonical_value(),
                "consequence": self.consequence.canonical_value(),
            }
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return None


class EvaluationMode(StrEnum):
    PROOF_JUDGED_V1 = "proof_judged_v1"


class ProofKind(StrEnum):
    GOLDEN = "golden"
    AGENT = "agent"


class JudgeVerdict(StrEnum):
    YES = "yes"
    NO = "no"
    INCONCLUSIVE = "inconclusive"


class GoldReferenceStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INCOMPLETE = "incomplete"
    CONTRADICTED = "contradicted"
    NOT_NEEDED = "not_needed"


class AnswerRelation(StrEnum):
    SAME_AS_GOLDEN = "same_as_golden"
    VALID_EXTENSION_OF_GOLDEN = "valid_extension_of_golden"
    VALID_ALTERNATIVE_TO_GOLDEN = "valid_alternative_to_golden"
    CONTRADICTS_GOLDEN_BUT_AGENT_PROOF_WINS = "contradicts_golden_but_agent_proof_wins"
    AGENT_ANSWER_INCORRECT = "agent_answer_incorrect"
    DATASET_ISSUE = "dataset_issue"


_ALTERNATIVE_ANSWER_RELATIONS = frozenset(
    {
        AnswerRelation.VALID_EXTENSION_OF_GOLDEN,
        AnswerRelation.VALID_ALTERNATIVE_TO_GOLDEN,
        AnswerRelation.CONTRADICTS_GOLDEN_BUT_AGENT_PROOF_WINS,
    }
)


def expected_valid_alternative_to_golden(relation: AnswerRelation) -> bool:
    return relation in _ALTERNATIVE_ANSWER_RELATIONS


class JudgeFailureCode(StrEnum):
    MISSING_AGENT_ANSWER = "missing_agent_answer"
    MISSING_AGENT_PROOF = "missing_agent_proof"
    AGENT_PROOF_EXECUTION_FAILED = "agent_proof_execution_failed"
    AGENT_PROOF_TIMED_OUT = "agent_proof_timed_out"
    AGENT_PROOF_MUTATED_REPOSITORY = "agent_proof_mutated_repository"
    AGENT_PROOF_OUTPUT_TRUNCATED = "agent_proof_output_truncated"
    AGENT_ANSWER_INCORRECT = "agent_answer_incorrect"
    AGENT_PROOF_ANSWER_MISMATCH = "agent_proof_answer_mismatch"
    AGENT_PROOF_INSUFFICIENT = "agent_proof_insufficient"
    GOLDEN_PROOF_EXECUTION_FAILED = "golden_proof_execution_failed"
    GOLDEN_PROOF_UNSUPPORTED = "golden_proof_unsupported"
    GOLD_REFERENCE_INCOMPLETE = "gold_reference_incomplete"
    GOLD_REFERENCE_CONTRADICTED = "gold_reference_contradicted"
    JUDGE_INCONCLUSIVE = "judge_inconclusive"
    INVALID_EVIDENCE = "invalid_evidence"


@dataclass(frozen=True, slots=True)
class GoldenAnswerReference:
    canonical: str

    def __post_init__(self) -> None:
        _required_string(self.canonical, "golden_answer.canonical")

    @classmethod
    def from_dict(cls, value: object) -> GoldenAnswerReference:
        if isinstance(value, str):
            return cls(canonical=value)
        if not isinstance(value, dict):
            raise TypeError("golden_answer must be a string or object")
        _strict_keys(value, {"canonical"}, set(), "golden_answer")
        return cls(canonical=value["canonical"])

    def to_dict(self) -> dict[str, str]:
        return {"canonical": self.canonical}


@dataclass(frozen=True, slots=True)
class GoldenProof:
    path: str
    sha256: str
    format: str = "bash"
    timeout_seconds: float = 30.0
    max_output_bytes: int = 65536

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "golden_proof.path", "golden-proofs")
        _sha256(self.sha256, "golden_proof.sha256")
        if self.format != "bash":
            raise ValueError("golden_proof.format must be bash")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("golden_proof.timeout_seconds must be greater than zero")
        if (
            not isinstance(self.max_output_bytes, int)
            or isinstance(self.max_output_bytes, bool)
            or self.max_output_bytes <= 0
        ):
            raise ValueError("golden_proof.max_output_bytes must be a positive integer")

    @classmethod
    def from_dict(cls, value: object) -> GoldenProof:
        if not isinstance(value, dict):
            raise TypeError("golden_proof must be an object")
        _strict_keys(
            value,
            {"path", "format", "sha256", "timeout_seconds", "max_output_bytes"},
            set(),
            "golden_proof",
        )
        return cls(
            path=value["path"],
            format=value["format"],
            sha256=value["sha256"],
            timeout_seconds=value["timeout_seconds"],
            max_output_bytes=value["max_output_bytes"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GoldenProofValidation:
    operationally_valid: bool
    repository_dependent: bool
    deterministic: bool
    repository_state_unchanged: bool
    # Whether the proof still holds once the issue is fixed. Optional because
    # reports predating the post-patch run carry no verdict either way, and
    # absence must not be read as failure.
    patch_stable: bool | None = None

    def __post_init__(self) -> None:
        for name in (
            "operationally_valid",
            "repository_dependent",
            "deterministic",
            "repository_state_unchanged",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.patch_stable is not None and not isinstance(self.patch_stable, bool):
            raise TypeError("patch_stable must be a boolean or absent")

    @classmethod
    def from_dict(cls, value: object) -> GoldenProofValidation:
        if not isinstance(value, dict):
            raise TypeError("golden_proof_validation must be an object")
        fields = {
            "operationally_valid",
            "repository_dependent",
            "deterministic",
            "repository_state_unchanged",
        }
        _strict_keys(value, fields, {"patch_stable"}, "golden_proof_validation")
        return cls(
            **{name: value[name] for name in fields},
            patch_stable=value.get("patch_stable"),
        )

    def to_dict(self) -> dict[str, bool]:
        value = asdict(self)
        if self.patch_stable is None:
            value.pop("patch_stable")
        return value


@dataclass(frozen=True, slots=True)
class ProofExecutionRecord:
    question_id: str
    proof_kind: ProofKind | str
    source_sha256: str
    submitted: bool
    executed: bool
    success: bool
    exit_code: int | None
    timed_out: bool
    output_truncated: bool
    repository_state_unchanged: bool
    evidence_valid: bool = True
    repository_dependent: bool | None = None
    deterministic: bool | None = None
    # True when the proof still succeeds after the issue's gold patch is applied.
    # None for reports written before the post-patch run existed, and for agent
    # proofs, which are not validated against the patched tree.
    patch_stable: bool | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    protocol: str = QA_PROOF_EXECUTION_PROTOCOL

    def __post_init__(self) -> None:
        _required_string(self.question_id, "question_id")
        object.__setattr__(self, "proof_kind", ProofKind(self.proof_kind))
        _sha256(self.source_sha256, "source_sha256")
        if self.protocol != QA_PROOF_EXECUTION_PROTOCOL:
            raise ValueError(f"protocol must be {QA_PROOF_EXECUTION_PROTOCOL}")
        for name in (
            "submitted",
            "executed",
            "success",
            "timed_out",
            "output_truncated",
            "repository_state_unchanged",
            "evidence_valid",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in (
            "repository_dependent",
            "deterministic",
            "patch_stable",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean or null")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise TypeError("exit_code must be an integer or null")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("stdout and stderr must be strings")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be non-negative")
        expected_success = (
            self.submitted
            and self.executed
            and self.exit_code == 0
            and not self.timed_out
            and not self.output_truncated
            and self.repository_state_unchanged
            and self.evidence_valid
        )
        if self.success != expected_success:
            raise ValueError("success must equal the trusted execution integrity result")

    @classmethod
    def from_dict(cls, value: object) -> ProofExecutionRecord:
        if not isinstance(value, dict):
            raise TypeError("Proof execution record must be an object")
        required = {
            "protocol",
            "question_id",
            "proof_kind",
            "source_sha256",
            "submitted",
            "executed",
            "success",
            "exit_code",
            "timed_out",
            "output_truncated",
            "repository_state_unchanged",
            "evidence_valid",
        }
        optional = {
            "stdout",
            "stderr",
            "duration_seconds",
            "repository_dependent",
            "deterministic",
            "patch_stable",
        }
        _strict_keys(value, required, optional, "proof execution record")
        return cls(
            question_id=value["question_id"],
            proof_kind=value["proof_kind"],
            source_sha256=value["source_sha256"],
            submitted=value["submitted"],
            executed=value["executed"],
            success=value["success"],
            exit_code=value["exit_code"],
            timed_out=value["timed_out"],
            output_truncated=value["output_truncated"],
            repository_state_unchanged=value["repository_state_unchanged"],
            evidence_valid=value["evidence_valid"],
            repository_dependent=value.get("repository_dependent"),
            deterministic=value.get("deterministic"),
            patch_stable=value.get("patch_stable"),
            stdout=value.get("stdout", ""),
            stderr=value.get("stderr", ""),
            duration_seconds=value.get("duration_seconds", 0.0),
            protocol=value["protocol"],
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["proof_kind"] = self.proof_kind.value
        if self.patch_stable is None:
            value.pop("patch_stable")
        return value


@dataclass(frozen=True, slots=True)
class ProofJudgeAssessment:
    question_id: str
    agent_answer_correct: JudgeVerdict | str
    agent_proof_matches_answer: JudgeVerdict | str
    agent_proof_establishes_answer: JudgeVerdict | str
    golden_proof_supports_golden_answer: JudgeVerdict | str
    gold_reference_status: GoldReferenceStatus | str
    valid_alternative_to_golden: bool
    relation: AnswerRelation | str
    confidence: float
    covered_claims: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    contradicted_claims: tuple[str, ...]
    failure_codes: tuple[JudgeFailureCode | str, ...]
    rationale: str
    protocol: str = QA_PROOF_JUDGE_PROTOCOL

    def __post_init__(self) -> None:
        _required_string(self.question_id, "question_id")
        if self.protocol != QA_PROOF_JUDGE_PROTOCOL:
            raise ValueError(f"protocol must be {QA_PROOF_JUDGE_PROTOCOL}")
        for name in (
            "agent_answer_correct",
            "agent_proof_matches_answer",
            "agent_proof_establishes_answer",
            "golden_proof_supports_golden_answer",
        ):
            object.__setattr__(self, name, JudgeVerdict(getattr(self, name)))
        object.__setattr__(
            self,
            "gold_reference_status",
            GoldReferenceStatus(self.gold_reference_status),
        )
        object.__setattr__(self, "relation", AnswerRelation(self.relation))
        object.__setattr__(
            self,
            "failure_codes",
            tuple(JudgeFailureCode(code) for code in self.failure_codes),
        )
        if not isinstance(self.valid_alternative_to_golden, bool):
            raise TypeError("valid_alternative_to_golden must be a boolean")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be between zero and one")
        for name in ("covered_claims", "unsupported_claims", "contradicted_claims"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a list")
            for claim in values:
                _required_string(claim, name)
        _required_string(self.rationale, "rationale")
        if self.valid_alternative_to_golden != expected_valid_alternative_to_golden(
            self.relation
        ):
            raise ValueError("valid_alternative_to_golden is inconsistent with relation")
        if self.relation is AnswerRelation.DATASET_ISSUE and self.gold_reference_status not in {
            GoldReferenceStatus.INCOMPLETE,
            GoldReferenceStatus.CONTRADICTED,
        }:
            raise ValueError("dataset_issue requires incomplete or contradicted gold")
        if (
            self.relation is AnswerRelation.AGENT_ANSWER_INCORRECT
            and self.agent_answer_correct is not JudgeVerdict.NO
        ):
            raise ValueError("agent_answer_incorrect requires agent_answer_correct=no")

    @property
    def dataset_issue(self) -> bool:
        return self.relation is AnswerRelation.DATASET_ISSUE

    @property
    def answer_correct(self) -> bool:
        return self.agent_answer_correct is JudgeVerdict.YES and not self.dataset_issue

    def answer_plus_proof_correct(
        self,
        execution: ProofExecutionRecord | None,
    ) -> bool:
        return (
            not self.dataset_issue
            and execution is not None
            and execution.success
            and self.agent_answer_correct is JudgeVerdict.YES
            and self.agent_proof_matches_answer is JudgeVerdict.YES
            and self.agent_proof_establishes_answer is JudgeVerdict.YES
        )

    @classmethod
    def from_dict(cls, value: object) -> ProofJudgeAssessment:
        if not isinstance(value, dict):
            raise TypeError("Judge assessment must be an object")
        required = {
            "protocol",
            "question_id",
            "agent_answer_correct",
            "agent_proof_matches_answer",
            "agent_proof_establishes_answer",
            "golden_proof_supports_golden_answer",
            "gold_reference_status",
            "valid_alternative_to_golden",
            "relation",
            "confidence",
            "covered_claims",
            "unsupported_claims",
            "contradicted_claims",
            "failure_codes",
            "rationale",
        }
        _strict_keys(value, required, set(), "judge assessment")
        list_fields = {
            name: value[name]
            for name in (
                "covered_claims",
                "unsupported_claims",
                "contradicted_claims",
                "failure_codes",
            )
        }
        if any(not isinstance(items, list) for items in list_fields.values()):
            raise TypeError("Judge claim and failure fields must be lists")
        payload = dict(value)
        relation = AnswerRelation(payload["relation"])
        expected = expected_valid_alternative_to_golden(relation)
        if payload["valid_alternative_to_golden"] != expected:
            payload["valid_alternative_to_golden"] = expected
        return cls(
            **{
                **payload,
                "covered_claims": tuple(payload["covered_claims"]),
                "unsupported_claims": tuple(payload["unsupported_claims"]),
                "contradicted_claims": tuple(payload["contradicted_claims"]),
                "failure_codes": tuple(payload["failure_codes"]),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "agent_answer_correct",
            "agent_proof_matches_answer",
            "agent_proof_establishes_answer",
            "golden_proof_supports_golden_answer",
            "gold_reference_status",
            "relation",
        ):
            value[name] = getattr(self, name).value
        value["failure_codes"] = [code.value for code in self.failure_codes]
        value["covered_claims"] = list(self.covered_claims)
        value["unsupported_claims"] = list(self.unsupported_claims)
        value["contradicted_claims"] = list(self.contradicted_claims)
        return value


@dataclass(frozen=True, slots=True)
class AnswerJudgeAssessment:
    question_id: str
    agent_answer_correct: JudgeVerdict | str
    golden_proof_supports_golden_answer: JudgeVerdict | str
    gold_reference_status: GoldReferenceStatus | str
    valid_alternative_to_golden: bool
    relation: AnswerRelation | str
    confidence: float
    covered_claims: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    contradicted_claims: tuple[str, ...]
    failure_codes: tuple[JudgeFailureCode | str, ...]
    rationale: str
    protocol: str = QA_ANSWER_JUDGE_PROTOCOL

    def __post_init__(self) -> None:
        _required_string(self.question_id, "question_id")
        if self.protocol != QA_ANSWER_JUDGE_PROTOCOL:
            raise ValueError(f"protocol must be {QA_ANSWER_JUDGE_PROTOCOL}")
        object.__setattr__(
            self,
            "agent_answer_correct",
            JudgeVerdict(self.agent_answer_correct),
        )
        object.__setattr__(
            self,
            "golden_proof_supports_golden_answer",
            JudgeVerdict(self.golden_proof_supports_golden_answer),
        )
        object.__setattr__(
            self,
            "gold_reference_status",
            GoldReferenceStatus(self.gold_reference_status),
        )
        object.__setattr__(self, "relation", AnswerRelation(self.relation))
        object.__setattr__(
            self,
            "failure_codes",
            tuple(JudgeFailureCode(code) for code in self.failure_codes),
        )
        if self.relation is AnswerRelation.CONTRADICTS_GOLDEN_BUT_AGENT_PROOF_WINS:
            raise ValueError("answer-only judge cannot select a proof-winning relation")
        if not isinstance(self.valid_alternative_to_golden, bool):
            raise TypeError("valid_alternative_to_golden must be a boolean")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be between zero and one")
        for name in ("covered_claims", "unsupported_claims", "contradicted_claims"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a list")
            for claim in values:
                _required_string(claim, name)
        _required_string(self.rationale, "rationale")
        if self.valid_alternative_to_golden != expected_valid_alternative_to_golden(
            self.relation
        ):
            raise ValueError(
                "valid_alternative_to_golden is inconsistent with relation"
            )
        if self.relation is AnswerRelation.DATASET_ISSUE and (
            self.gold_reference_status
            not in {
                GoldReferenceStatus.INCOMPLETE,
                GoldReferenceStatus.CONTRADICTED,
            }
        ):
            raise ValueError(
                "dataset_issue requires incomplete or contradicted gold"
            )
        if (
            self.relation is AnswerRelation.AGENT_ANSWER_INCORRECT
            and self.agent_answer_correct is not JudgeVerdict.NO
        ):
            raise ValueError(
                "agent_answer_incorrect requires agent_answer_correct=no"
            )

    @property
    def dataset_issue(self) -> bool:
        return self.relation is AnswerRelation.DATASET_ISSUE

    @property
    def answer_correct(self) -> bool:
        return (
            self.agent_answer_correct is JudgeVerdict.YES
            and not self.dataset_issue
        )

    @classmethod
    def from_dict(cls, value: object) -> AnswerJudgeAssessment:
        if not isinstance(value, dict):
            raise TypeError("Answer judge assessment must be an object")
        required = {
            "protocol",
            "question_id",
            "agent_answer_correct",
            "golden_proof_supports_golden_answer",
            "gold_reference_status",
            "valid_alternative_to_golden",
            "relation",
            "confidence",
            "covered_claims",
            "unsupported_claims",
            "contradicted_claims",
            "failure_codes",
            "rationale",
        }
        _strict_keys(value, required, set(), "answer judge assessment")
        list_fields = {
            name: value[name]
            for name in (
                "covered_claims",
                "unsupported_claims",
                "contradicted_claims",
                "failure_codes",
            )
        }
        if any(not isinstance(items, list) for items in list_fields.values()):
            raise TypeError("Answer judge claim and failure fields must be lists")
        payload = dict(value)
        relation = AnswerRelation(payload["relation"])
        expected = expected_valid_alternative_to_golden(relation)
        if payload["valid_alternative_to_golden"] != expected:
            payload["valid_alternative_to_golden"] = expected
        return cls(
            **{
                **payload,
                "covered_claims": tuple(payload["covered_claims"]),
                "unsupported_claims": tuple(payload["unsupported_claims"]),
                "contradicted_claims": tuple(payload["contradicted_claims"]),
                "failure_codes": tuple(payload["failure_codes"]),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "agent_answer_correct",
            "golden_proof_supports_golden_answer",
            "gold_reference_status",
            "relation",
        ):
            value[name] = getattr(self, name).value
        value["failure_codes"] = [
            code.value for code in self.failure_codes
        ]
        value["covered_claims"] = list(self.covered_claims)
        value["unsupported_claims"] = list(self.unsupported_claims)
        value["contradicted_claims"] = list(self.contradicted_claims)
        return value


@dataclass(frozen=True, slots=True)
class QARecord:
    instance_id: str
    fact_id: str
    fact: str
    question_id: str
    question: str
    accepted_answers: tuple[str, ...] = ()
    incorrect_answers: tuple[str, ...] = ()
    probe_path: str | None = None
    answer_contract: AnswerContract | None = None
    category: QuestionCategory | str | None = None
    flow_scope: FlowScope | None = None
    evidence_hops: tuple[EvidenceHop, ...] = ()
    causal_chain: tuple[CausalStep, ...] = ()
    adversarial_review: AdversarialReview | None = None
    probe_validation: ProbeValidation | None = None
    golden_answer: GoldenAnswerReference | None = None
    golden_proof: GoldenProof | None = None
    golden_proof_validation: GoldenProofValidation | None = None
    evaluation_mode: EvaluationMode | str | None = None
    schema_version: str = QA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _validate_schema_version(self.schema_version),
        )
        for name in ("instance_id", "fact_id", "fact", "question_id", "question"):
            _required_string(getattr(self, name), name)
        if self.schema_version == QA_SCHEMA_V2:
            self._validate_v2()
        else:
            self._validate_v1()
        if self.category is not None:
            object.__setattr__(self, "category", QuestionCategory(self.category))
        if self.evidence_hops or self.causal_chain:
            if len(self.evidence_hops) < 3:
                raise ValueError("Complex questions require at least three evidence hops")
            if len({hop.file for hop in self.evidence_hops}) < 2:
                raise ValueError("Complex questions require evidence from at least two files")
            evidence_ids = [hop.evidence_id for hop in self.evidence_hops]
            if len(set(evidence_ids)) != len(evidence_ids):
                raise ValueError("Evidence identifiers must be unique")
            if len(self.causal_chain) < 3:
                raise ValueError("Complex questions require at least three causal steps")
            step_ids = [step.step_id for step in self.causal_chain]
            if len(set(step_ids)) != len(step_ids):
                raise ValueError("Causal step identifiers must be unique")
            unknown_evidence = {
                evidence_id
                for step in self.causal_chain
                for evidence_id in step.evidence_ids
                if evidence_id not in set(evidence_ids)
            }
            if unknown_evidence:
                raise ValueError(
                    f"Causal chain references unknown evidence: {sorted(unknown_evidence)}"
                )

    def _validate_v1(self) -> None:
        if not self.accepted_answers:
            raise ValueError("accepted_answers must contain at least one answer")
        if len(self.incorrect_answers) != 3:
            raise ValueError("incorrect_answers must contain exactly three answers")
        all_answers = (*self.accepted_answers, *self.incorrect_answers)
        if any(not isinstance(answer, str) or "\n" in answer or "\r" in answer for answer in all_answers):
            raise ValueError("Answers must be single-line strings")
        normalized_accepted = [normalize_answer(answer) for answer in self.accepted_answers]
        normalized_incorrect = [normalize_answer(answer) for answer in self.incorrect_answers]
        if any(not answer for answer in (*normalized_accepted, *normalized_incorrect)):
            raise ValueError("Answers must not be empty")
        if len(set(normalized_incorrect)) != 3:
            raise ValueError("incorrect_answers must be unique after normalization")
        if set(normalized_accepted) & set(normalized_incorrect):
            raise ValueError("Accepted and incorrect answers must not overlap")
        probe_path = PurePosixPath(_required_string(self.probe_path, "probe_path"))
        if probe_path.is_absolute() or ".." in probe_path.parts:
            raise ValueError("probe_path must be a safe relative path")
        expected_parent = PurePosixPath("probes") / self.instance_id
        if probe_path.parent != expected_parent or not probe_path.name.startswith("q-") or probe_path.suffix != ".sh":
            raise ValueError(f"probe_path must match probes/{self.instance_id}/q-*.sh")
        if (
            self.answer_contract is not None
            and self.answer_contract.kind
            in {
                AnswerContractKind.CAUSAL_FLOW_V1,
                AnswerContractKind.RATIONALE_V1,
            }
            and self.flow_scope is None
        ):
            raise ValueError("Structured deterministic questions require flow_scope")
        if any(
            value is not None
            for value in (
                self.golden_answer,
                self.golden_proof,
                self.golden_proof_validation,
                self.evaluation_mode,
            )
        ):
            raise ValueError("v1 records do not accept proof evaluation fields")

    def _validate_v2(self) -> None:
        if self.category is None:
            raise ValueError("v2 records require category")
        if len(self.evidence_hops) < 3:
            raise ValueError("v2 records require at least three evidence hops")
        if len(self.causal_chain) < 3:
            raise ValueError("v2 records require at least three reasoning steps")
        if self.golden_answer is None:
            raise ValueError("v2 records require golden_answer")
        if self.golden_proof is None:
            raise ValueError("v2 records require golden_proof")
        if self.evaluation_mode is None:
            raise ValueError("v2 records require evaluation_mode")
        object.__setattr__(self, "evaluation_mode", EvaluationMode(self.evaluation_mode))
        if self.accepted_answers or self.incorrect_answers or self.probe_path is not None:
            raise ValueError("v2 proof-judged records do not accept v1 oracle fields")
        if self.answer_contract is not None or self.probe_validation is not None:
            raise ValueError("v2 proof-judged records do not accept deterministic validation fields")

    @classmethod
    def from_dict(cls, value: dict[str, Any], default_instance_id: str | None = None) -> QARecord:
        validation = value.get("probe_validation")
        proof_validation = value.get("golden_proof_validation")
        review = value.get("adversarial_review")
        contract = value.get("answer_contract")
        category = value.get("category")
        flow_scope = value.get("flow_scope")
        return cls(
            instance_id=value.get("instance_id", default_instance_id),
            fact_id=value.get("fact_id"),
            fact=value.get("fact"),
            question_id=value.get("question_id"),
            question=value.get("question"),
            accepted_answers=tuple(value.get("accepted_answers", ())),
            incorrect_answers=tuple(value.get("incorrect_answers", ())),
            probe_path=value.get("probe_path"),
            answer_contract=AnswerContract.from_dict(contract) if contract is not None else None,
            category=category,
            flow_scope=(
                FlowScope.from_dict(flow_scope) if flow_scope is not None else None
            ),
            evidence_hops=tuple(
                EvidenceHop.from_dict(item) for item in value.get("evidence_hops", ())
            ),
            causal_chain=tuple(
                CausalStep.from_dict(item)
                for item in value.get(
                    "causal_chain",
                    value.get("reasoning_chain", ()),
                )
            ),
            adversarial_review=(
                AdversarialReview.from_dict(review) if review is not None else None
            ),
            probe_validation=ProbeValidation.from_dict(validation) if validation is not None else None,
            golden_answer=(
                GoldenAnswerReference.from_dict(value["golden_answer"])
                if value.get("golden_answer") is not None
                else None
            ),
            golden_proof=(
                GoldenProof.from_dict(value["golden_proof"])
                if value.get("golden_proof") is not None
                else None
            ),
            golden_proof_validation=(
                GoldenProofValidation.from_dict(proof_validation)
                if proof_validation is not None
                else None
            ),
            evaluation_mode=value.get("evaluation_mode"),
            schema_version=value.get("schema_version", QA_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["accepted_answers"] = list(self.accepted_answers)
        value["incorrect_answers"] = list(self.incorrect_answers)
        if self.answer_contract is not None:
            value["answer_contract"] = self.answer_contract.to_dict()
        if self.category is not None:
            value["category"] = self.category.value
        if self.flow_scope is not None:
            value["flow_scope"] = self.flow_scope.to_dict()
        value["evidence_hops"] = [hop.to_dict() for hop in self.evidence_hops]
        value["causal_chain"] = [step.to_dict() for step in self.causal_chain]
        if self.schema_version == QA_SCHEMA_V2:
            value["reasoning_chain"] = value.pop("causal_chain")
        if self.adversarial_review is not None:
            value["adversarial_review"] = self.adversarial_review.to_dict()
        if self.probe_validation is not None:
            value["probe_validation"] = self.probe_validation.to_dict()
        if self.golden_answer is not None:
            value["golden_answer"] = self.golden_answer.to_dict()
        if self.golden_proof is not None:
            value["golden_proof"] = self.golden_proof.to_dict()
        if self.golden_proof_validation is not None:
            value["golden_proof_validation"] = self.golden_proof_validation.to_dict()
        else:
            value.pop("golden_proof_validation")
        if self.evaluation_mode is not None:
            value["evaluation_mode"] = self.evaluation_mode.value
        return value


Question = QARecord


@dataclass(frozen=True, slots=True)
class GenerationBundle:
    instance_id: str
    documentation: str
    facts: tuple[Fact, ...]
    questions: tuple[QARecord, ...]
    probes: dict[str, str]
    golden_proofs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    schema_version: str = QA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _validate_schema_version(self.schema_version),
        )
        _required_string(self.instance_id, "instance_id")
        _required_string(self.documentation, "documentation")
        if not self.facts:
            raise ValueError("facts must not be empty")
        if not self.questions:
            raise ValueError("questions must not be empty")
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("Fact identifiers must be unique")
        question_ids = [question.question_id for question in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Question identifiers must be unique")
        question_fact_ids = [question.fact_id for question in self.questions]
        if len(set(question_fact_ids)) != len(question_fact_ids):
            raise ValueError("Each fact may be tested by at most one question")
        facts_by_id = {fact.fact_id: fact for fact in self.facts}
        for question in self.questions:
            fact = facts_by_id.get(question.fact_id)
            if fact is None:
                raise ValueError(f"Unknown fact_id for {question.question_id}: {question.fact_id}")
            if question.fact != fact.fact:
                raise ValueError(f"Fact text mismatch for {question.question_id}")
            if question.instance_id != self.instance_id or fact.instance_id != self.instance_id:
                raise ValueError("All records must use the bundle instance_id")
            if question.schema_version != self.schema_version:
                raise ValueError("All records must use the bundle schema_version")
            if self.schema_version == QA_SCHEMA_VERSION:
                if question.probe_path not in self.probes:
                    raise ValueError(f"Missing probe script: {question.probe_path}")
            elif question.golden_proof.path not in self.golden_proofs:
                raise ValueError(f"Missing golden proof: {question.golden_proof.path}")
        if self.schema_version == QA_SCHEMA_VERSION and self.golden_proofs:
            raise ValueError("v1 bundles do not accept golden proofs")
        if self.schema_version == QA_SCHEMA_V2:
            expected = {question.golden_proof.path for question in self.questions}
            if set(self.golden_proofs) != expected:
                raise ValueError("Golden proof paths must map one-to-one to questions")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GenerationBundle:
        instance_id = value.get("instance_id")
        schema_version = _validate_schema_version(
            value.get("schema_version", QA_SCHEMA_VERSION)
        )
        facts = tuple(
            Fact.from_dict(item, instance_id, schema_version)
            for item in value.get("facts", ())
        )
        facts_by_id = {fact.fact_id: fact for fact in facts}
        questions = []
        probes = dict(value.get("probes", {}))
        golden_proofs = dict(value.get("golden_proofs", {}))
        for item in value.get("questions", ()):
            record = dict(item)
            record.setdefault("schema_version", schema_version)
            fact = facts_by_id.get(record.get("fact_id"))
            if fact is not None:
                record.setdefault("fact", fact.fact)
            record.setdefault("instance_id", instance_id)
            if _validate_schema_version(schema_version) == QA_SCHEMA_VERSION:
                record.setdefault(
                    "probe_path",
                    f"probes/{instance_id}/{record.get('question_id')}.sh",
                )
            script = record.pop("probe", None)
            if script is not None:
                probes.setdefault(record["probe_path"], script)
            proof_script = record.pop("golden_proof_script", None)
            proof_metadata = record.get("golden_proof")
            if proof_script is not None and isinstance(proof_metadata, dict):
                golden_proofs.setdefault(proof_metadata.get("path"), proof_script)
            questions.append(QARecord.from_dict(record, instance_id))
        return cls(
            instance_id=instance_id,
            documentation=_documentation_text(value.get("documentation")),
            facts=facts,
            questions=tuple(questions),
            probes=probes,
            golden_proofs=golden_proofs,
            metadata=dict(value.get("metadata", {})),
            debug=dict(value.get("debug", {})),
            schema_version=value.get("schema_version", QA_SCHEMA_VERSION),
        )
