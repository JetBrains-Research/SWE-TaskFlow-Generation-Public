import json

from interactive_benchmark_generation.prompts import build_check_prompt, build_filter_prompt, build_split_prompt


def test_build_filter_prompt_rejects_evidence_only_atomic_tasks():
    prompt = build_filter_prompt("A single behavior is broken.")

    assert "at least two source-grounded units of work" in prompt
    assert "atomic_evidence_only" in prompt
    assert "Multiple examples showing the same expected behavior" in prompt
    assert "source_evidence" in prompt


def test_build_split_prompt_contains_schema_and_problem_statement():
    prompt = build_split_prompt(
        "Bug happens when nested models are composed.\n[[VERBATIM_ARTIFACT_1]]",
        3,
        previous_issues=["Part 2 is missing the artifact."],
        verbatim_placeholders=["[[VERBATIM_ARTIFACT_1]]"],
        incremental_units=["Support nested composition.", "Preserve evaluation order."],
        previous_parts=[
            "Fix nested model composition.",
            "[[VERBATIM_ARTIFACT_1]]",
            "Preserve evaluation order.",
        ],
    )

    assert "exactly n_parts" in prompt
    assert "This is not a paragraph split" in prompt
    assert "Stage 1: decompose the source" in prompt
    assert "Stage 2: compose the user messages" in prompt
    assert "A specificity unit is" in prompt
    assert "A subtask unit is" in prompt
    assert "not required to disclose every requirement or subtask" in prompt
    assert "Preserve exact technical artifacts" in prompt
    assert "never include both" in prompt
    assert "Do not create duplicate units" in prompt
    assert "Assign each detailed Stage 1 unit to exactly one part" in prompt
    assert "necessary high-level-to-contract progression is not duplication" in prompt
    assert "completely and verbatim" in prompt
    assert "Include every placeholder exactly once" in prompt
    assert '"verbatim_placeholders": [' in prompt
    assert '"incremental_units": [' in prompt
    assert '"previous_parts": [' in prompt
    assert "[[VERBATIM_ARTIFACT_1]]" in prompt
    assert "with verbatim placeholders still unexpanded" in prompt
    assert "substantive task information" in prompt
    assert "Bug happens when nested models are composed." in prompt
    assert '"n_parts": 3' in prompt
    assert '"specificity_units"' in prompt
    assert '"subtask_units"' in prompt
    assert '"parts"' in prompt


def test_build_check_prompt_embeds_parts_as_json():
    prompt = build_check_prompt("Original", ["First", "Second"])
    payload_text = prompt.split("Input:\n", 1)[1]
    payload = json.loads(payload_text)

    assert "paragraph-based" in prompt
    assert "increased specificity, an additional requested subtask, or both" in prompt
    assert "Exact technical artifacts" in prompt
    assert "every distinct piece of information" in prompt
    assert "No prose fact is presented both as a paraphrase" in prompt
    assert "do not repeat concrete acceptance criteria" in prompt
    assert "Do not reject this high-level-to-contract progression as duplication" in prompt
    assert "Representative samples" in prompt
    assert "substantive task information" in prompt
    assert "Independently reassess problem_statement" in prompt
    assert "Do not trust incremental_units as proof" in prompt
    assert payload["problem_statement"] == "Original"
    assert payload["parts"] == ["First", "Second"]
