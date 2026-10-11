import re
from dataclasses import dataclass

LABEL_RE = re.compile(r"^\s*(part|message|prompt)\s*\d+\s*[:.)-]\s*", re.IGNORECASE)
TRACEBACK_RE = re.compile(r"traceback|stack trace|^\s*file .+, line \d+", re.IGNORECASE | re.MULTILINE)
TECHNICAL_MARKER_RE = re.compile(r"`|```|[./][\w.-]+|\w+\(|\w+__\w+")
INTERFACE_SECTION_RE = re.compile(r"(?P<artifact>^## Interface[ \t]*\n.*\Z)", re.MULTILINE | re.DOTALL)
FENCED_BLOCK_RE = re.compile(
    r"(?P<artifact>^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^(?P=indent)(?P=fence)[ \t]*$)",
    re.MULTILINE | re.DOTALL,
)
UNFENCED_TRACEBACK_RE = re.compile(
    r"(?P<artifact>^Traceback \(most recent call last\):\n"
    r"(?:(?:[ \t]+.*|During handling of the above exception.*|The above exception was.*)\n)*"
    r"^[A-Za-z_][\w.]*?(?:Error|Exception):.*$)",
    re.MULTILINE,
)
INDENTED_BLOCK_RE = re.compile(r"(?P<artifact>(?:^(?: {4}|\t).*(?:\n|$)){2,})", re.MULTILINE)
WORD_RE = re.compile(r"\w+")
LOW_INFORMATION_WORDS = {
    "a",
    "am",
    "an",
    "and",
    "be",
    "bug",
    "but",
    "does",
    "expected",
    "feel",
    "feeling",
    "feels",
    "i",
    "is",
    "it",
    "like",
    "maybe",
    "me",
    "missing",
    "might",
    "or",
    "seems",
    "something",
    "that",
    "the",
    "this",
    "to",
}


@dataclass(frozen=True)
class LocalValidationResult:
    ok: bool
    issues: list[str]


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def substantive_word_count(text: str) -> int:
    return sum(1 for word in WORD_RE.findall(text.lower()) if word not in LOW_INFORMATION_WORDS)


def extract_verbatim_artifacts(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = []
    occupied_ranges = []
    for pattern in (INTERFACE_SECTION_RE, FENCED_BLOCK_RE, UNFENCED_TRACEBACK_RE, INDENTED_BLOCK_RE):
        for match in pattern.finditer(text):
            start, end = match.span("artifact")
            if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied_ranges):
                continue
            matches.append((start, match.group("artifact")))
            occupied_ranges.append((start, end))
    return [artifact for _start, artifact in sorted(matches)]


def artifact_excerpt(artifact: str, limit: int = 120) -> str:
    excerpt = " ".join(artifact.split())
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit - 3] + "..."


def validate_verbatim_placeholders(parts: list[str], placeholders: list[str]) -> LocalValidationResult:
    issues = []
    for placeholder in placeholders:
        count = sum(part.count(placeholder) for part in parts)
        if count != 1:
            issues.append(f"Placeholder {placeholder} must appear exactly once, found {count}")
    return LocalValidationResult(ok=not issues, issues=issues)


def validate_parts_locally(problem_statement: str, parts: list[str], n_parts: int) -> LocalValidationResult:
    issues = []
    if len(parts) != n_parts:
        issues.append(f"Expected {n_parts} parts, got {len(parts)}")

    normalized_original = normalize_text(problem_statement)
    seen = set()

    for index, part in enumerate(parts, start=1):
        normalized_part = normalize_text(part)
        if not normalized_part:
            issues.append(f"Part {index} is empty")
            continue
        if len(WORD_RE.findall(part)) < 4:
            issues.append(f"Part {index} is too short")
        if index > 1 and substantive_word_count(part) <= 3 and not TECHNICAL_MARKER_RE.search(part):
            issues.append(f"Part {index} does not contain enough substantive task information")
        if LABEL_RE.match(part):
            issues.append(f"Part {index} starts with a label")
        if normalized_part in seen:
            issues.append(f"Part {index} duplicates an earlier part")
        seen.add(normalized_part)

    if parts:
        first = parts[0]
        first_words = len(WORD_RE.findall(first))
        original_words = max(len(WORD_RE.findall(problem_statement)), 1)
        if first_words < min(30, max(8, original_words // 10)):
            issues.append("First part is too short to be independently actionable")
        if TRACEBACK_RE.search(first) and first_words < 60:
            issues.append("First part looks like diagnostics without a task description")

    joined = normalize_text("\n\n".join(parts))
    if normalized_original and not joined:
        issues.append("Joined parts are empty")

    merged_parts = "\n\n".join(parts).replace("\r\n", "\n")
    for index, artifact in enumerate(extract_verbatim_artifacts(problem_statement), start=1):
        normalized_artifact = artifact.replace("\r\n", "\n")
        if normalized_artifact not in merged_parts:
            issues.append(
                f"Verbatim artifact {index} is missing or changed: {artifact_excerpt(artifact)!r}. "
                "Copy the complete block from problem_statement exactly into one of the parts without shortening, "
                "summarizing, or reformatting it."
            )

    return LocalValidationResult(ok=not issues, issues=issues)
