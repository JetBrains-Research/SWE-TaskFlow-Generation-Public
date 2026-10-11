from interactive_benchmark_generation.validation import validate_parts_locally, validate_verbatim_placeholders


def test_validate_parts_locally_accepts_useful_parts():
    original = (
        "A library fails when composing nested models in arithmetic expressions. "
        "The issue appears during evaluation and the expected behavior is that nested models compose correctly. "
        "A stack trace points to the model composition helper."
    )
    parts = [
        "A library fails when composing nested models in arithmetic expressions. The expected behavior is that nested models compose correctly.",
        "The issue appears during evaluation.",
        "A stack trace points to the model composition helper.",
    ]

    result = validate_parts_locally(original, parts, 3)

    assert result.ok
    assert result.issues == []


def test_validate_parts_locally_rejects_empty_and_labeled_parts():
    original = "A bug appears when rendering a chart with nested data."
    parts = ["Part 1: A bug appears when rendering a chart with nested data.", "", "extra"]

    result = validate_parts_locally(original, parts, 3)

    assert not result.ok
    assert any("starts with a label" in issue for issue in result.issues)
    assert any("empty" in issue for issue in result.issues)


def test_validate_parts_locally_rejects_low_information_part():
    original = (
        "Nested compound models produce a separability matrix that marks independent linear models as dependent. "
        "The same linear models are separable when they are not nested."
    )
    parts = [
        "Nested compound models produce a separability matrix that marks independent linear models as dependent.",
        "The same linear models are separable when they are not nested.",
        "This feels like a bug to me, but I might be missing something?",
    ]

    result = validate_parts_locally(original, parts, 3)

    assert not result.ok
    assert any("substantive task information" in issue for issue in result.issues)


def test_validate_parts_locally_requires_complete_fenced_blocks():
    original = (
        "Pylint checks generated files unexpectedly.\n\n"
        "```shell\n"
        "src/gen/about.py:2:0: R2044: Line with empty comment\n"
        "src/gen/design.py:12:0: R0902: Too many instance attributes\n"
        "```\n\n"
        "Generated files should be ignored."
    )
    parts = [
        "Pylint checks generated files unexpectedly, although generated files should be ignored.",
        "The output includes `src/gen/about.py` and `src/gen/design.py`.",
        "It reports R2044 and R0902 errors from those generated files.",
    ]

    result = validate_parts_locally(original, parts, 3)

    assert not result.ok
    assert any("Verbatim artifact 1 is missing or changed" in issue for issue in result.issues)


def test_validate_parts_locally_accepts_complete_fenced_blocks():
    artifact = (
        "```shell\n"
        "src/gen/about.py:2:0: R2044: Line with empty comment\n"
        "src/gen/design.py:12:0: R0902: Too many instance attributes\n"
        "```"
    )
    original = f"Pylint checks generated files unexpectedly.\n\n{artifact}\n\nGenerated files should be ignored."
    parts = [
        "Pylint checks generated files unexpectedly, although generated files should be ignored.",
        f"Here is the complete output:\n\n{artifact}",
        "The affected target is the generated files under src/gen.",
    ]

    result = validate_parts_locally(original, parts, 3)

    assert result.ok
    assert result.issues == []


def test_validate_parts_locally_requires_complete_interface_section():
    original = (
        "Implement YAML-native export with editable attachments.\n\n"
        "## Requirements\n\n"
        "Export nested attachments and preserve null values.\n\n"
        "## Interface\n\n"
        "Add `Exporter.Export(ctx context.Context, writer io.Writer) error` in `internal/ext/exporter.go`."
    )
    parts = [
        "Implement YAML-native export with editable attachments and preserve nested data.",
        "Export nested attachments while preserving null values.",
        "Add an Export method to the Exporter type.",
    ]

    result = validate_parts_locally(original, parts, 3)

    assert not result.ok
    assert any("## Interface" in issue for issue in result.issues)


def test_validate_verbatim_placeholders_accepts_each_placeholder_once():
    result = validate_verbatim_placeholders(
        ["Initial request.", "[[VERBATIM_ARTIFACT_1]]", "[[VERBATIM_INTERFACE_SECTION]]"],
        ["[[VERBATIM_ARTIFACT_1]]", "[[VERBATIM_INTERFACE_SECTION]]"],
    )

    assert result.ok
    assert result.issues == []


def test_validate_verbatim_placeholders_rejects_missing_and_duplicate_placeholders():
    result = validate_verbatim_placeholders(
        [
            "[[VERBATIM_ARTIFACT_1]]",
            "Again: [[VERBATIM_ARTIFACT_1]]",
        ],
        ["[[VERBATIM_ARTIFACT_1]]", "[[VERBATIM_INTERFACE_SECTION]]"],
    )

    assert not result.ok
    assert "Placeholder [[VERBATIM_ARTIFACT_1]] must appear exactly once, found 2" in result.issues
    assert "Placeholder [[VERBATIM_INTERFACE_SECTION]] must appear exactly once, found 0" in result.issues
