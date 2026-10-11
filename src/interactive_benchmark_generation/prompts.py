import json

SPLIT_SYSTEM_PROMPT = """You decompose software benchmark issue prompts into source-grounded units and then transform those units into a sequence of natural user messages for an interactive coding agent.

You must preserve the original information only. Do not add file paths, line numbers, stack frames, root causes, implementation details, tests, or advice that is not already present in the original prompt.

Do not split by paragraphs. Build a logical progression in which later messages may increase the specificity of the task, introduce additional source-grounded subtasks, or do both.

Natural user messages may contain pasted code, complete stack traces, command output, logs, and other long technical artifacts. Natural phrasing applies only to conversational prose and ordering. Technical artifacts must never be shortened, sampled, summarized, or rewritten. Prose may be rephrased once when all unique information is retained.

Preserve every distinct piece of original information, but do not repeat semantically equivalent information merely because the source states it more than once. Merge overlapping source statements while retaining the union of their unique details. In particular, code snippets, exact expected or actual outputs, stack traces, error messages, constraints, identifiers, and negative requirements from the original prompt must still appear somewhere across the final parts. Copy fenced code blocks, stack traces, command output, logs, and other verbatim artifacts completely and exactly.

Return strict JSON only."""


CHECK_SYSTEM_PROMPT = """You validate dataset transformations.

Return strict JSON only."""


FILTER_SYSTEM_PROMPT = """You classify whether software benchmark issues support incremental interactive requirements.

Use only information explicitly present in the original issue. Return strict JSON only."""


def build_filter_prompt(problem_statement: str, previous_issues: list[str] | None = None) -> str:
    payload = {
        "problem_statement": problem_statement,
        "previous_issues": previous_issues or [],
        "required_json_schema": {
            "eligible": "boolean",
            "category": "atomic_evidence_only | incremental_requirements | composite_subtasks",
            "units": [
                {
                    "requirement": "string",
                    "source_evidence": "exact quote from problem_statement",
                }
            ],
            "reason": "string",
        },
    }
    return f"""Classify whether problem_statement contains at least two source-grounded units of work that can be revealed incrementally.

Accept:
- incremental_requirements: one overall fix or feature with at least two distinct requirements, constraints, behaviors, or edge cases that independently change its acceptance criteria.
- composite_subtasks: at least two separately implementable changes explicitly requested by the issue.
- In SWE-bench Pro, a nontrivial Interface section may be a distinct unit when it adds exact public names, signatures, return types, or locations required by tests beyond the behavioral requirements.

Reject as atomic_evidence_only when the issue asks for only one behavioral change and the remaining content merely demonstrates, diagnoses, motivates, or restates it.

A unit of work must:
- Be explicitly present in problem_statement.
- Describe requested behavior, a required constraint, a distinct edge case, or a separately implementable change.
- Remain meaningful if another unit is omitted.
- Add a distinct acceptance condition rather than more evidence for the same condition.

These never count as additional units:
- A reproduction, stack trace, command output, log, version, environment, screenshot, or diagnostic.
- Multiple examples showing the same expected behavior.
- Expected and actual output that only describe one behavioral change.
- A paraphrase, restated request, uncertainty, question about whether behavior is intended, ease-of-fix assessment, politeness, or closing remark.
- A broad goal and a narrower restatement of that same goal.
- The boilerplate statement "No new interfaces are introduced."
- Multiple requirement or interface entries that only apply the same behavior to different input variants, container types, classes, methods, or execution paths.

Set eligible=true only for incremental_requirements or composite_subtasks with at least two valid units. For every unit, source_evidence must be an exact non-empty quote from problem_statement that proves the unit is source-grounded. Do not infer or invent requirements.

If previous_issues is non-empty, correct those exact classification problems.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""


def build_split_prompt(
    problem_statement: str,
    n_parts: int,
    previous_issues: list[str] | None = None,
    verbatim_placeholders: list[str] | None = None,
    incremental_units: list[str] | None = None,
    previous_parts: list[str] | None = None,
) -> str:
    payload = {
        "n_parts": n_parts,
        "problem_statement": problem_statement,
        "previous_issues": previous_issues or [],
        "previous_parts": previous_parts or [],
        "verbatim_placeholders": verbatim_placeholders or [],
        "incremental_units": incremental_units or [],
        "required_json_schema": {
            "specificity_units": [
                {
                    "content": "string",
                    "source_evidence": "exact quote from problem_statement",
                }
            ],
            "subtask_units": [
                {
                    "content": "string",
                    "source_evidence": "exact quote from problem_statement",
                }
            ],
            "parts": ["string"],
            "rationale": "string",
            "coverage_check": "string",
        },
    }
    return f"""Split the problem_statement into exactly n_parts user messages.

Perform two stages in one response. Complete the decomposition fields first, then use that decomposition to produce parts. This is one model response, not two separate requests.

Stage 1: decompose the source
- A specificity unit is a source-grounded piece of information that makes the same task more determined. Specificity units include the core goal, target, context, requirements, constraints, behavior, examples, expected or actual results, edge cases, diagnostics, implementation guidance, negative requirements, interfaces, and other exact artifacts.
- A subtask unit is a separately implementable change explicitly requested by the source. It remains a meaningful requested change if another subtask is omitted.
- Requirements belong in specificity_units unless they request separately implementable changes. Do not duplicate a unit across both lists.
- Decompose all distinct information in problem_statement, not only the most important information. Merge semantically equivalent or overlapping source statements into one unit while retaining every unique qualifier, constraint, identifier, and example. Do not create duplicate units. Every unit must be supported by an exact, non-empty source_evidence quote.
- incremental_units contains candidates extracted during an earlier eligibility classification. It is supporting input, not a third output unit type. Verify every candidate against problem_statement and classify it as either a specificity unit or a subtask unit. Do not assume that a candidate is valid merely because it appears in incremental_units.
- A reproduction, stack trace, command output, log, environment detail, or example is normally a specificity unit, not a subtask.
- Multiple examples of the same requested behavior are separate specificity evidence only when preserving them separately is necessary to retain all source information. They are not separate subtasks.

Stage 2: compose the user messages
- This is not a paragraph split. Combine the Stage 1 units into a realistic sequence of user requests.
- Later messages may add previously undisclosed specificity units, previously undisclosed subtask units, or both.
- The sequence does not need to make parts[0] sufficient to satisfy every acceptance criterion or pass the final tests. Requirements, interfaces, constraints, edge cases, and entire subtasks needed by the final tests may be introduced later.
- You may rephrase prose only when no unique information is lost. For each prose fact, choose either a faithful rephrasing or the original wording; never include both. Do not add a summary immediately before or after the same information in source wording.

Invariants:
- parts[0] must express the main idea and core goal of the original task. It must contain enough target and context for an agent to complete a meaningful, coherent initial implementation or fix.
- parts[0] is not required to disclose every requirement or subtask, describe the complete final behavior, or be sufficient to pass the final test suite.
- parts[0] may include one compact concrete symptom or expected-vs-actual contrast if the task would otherwise be impossible to localize.
- parts[0] must not dump the full original prompt, all examples, all stack traces, or every expected/actual output.
- Every later part must add at least one useful Stage 1 unit that was not disclosed earlier. It may refine the existing task, expand its acceptance criteria, introduce another requested subtask, or combine these forms of progression.
- Distribute both specificity_units and subtask_units progressively. Do not collapse all source-grounded requirements or subtasks into parts[0].
- Assign each detailed Stage 1 unit to exactly one part. Do not restate concrete acceptance criteria, constraints, examples, diagnostics, or subtasks already represented in another part.
- A broad core goal in parts[0] may overlap with a later verbatim Interface when the Interface adds the exact API name, signature, type, return value, or location. This necessary high-level-to-contract progression is not duplication. Do not copy or paraphrase the detailed Interface contract outside its placeholder.
- No part may be empty or filler.
- Every part must contain substantive task information. Do not create a part whose only contribution is uncertainty, politeness, frustration, meta-commentary, or a closing remark.
- Do not put nearly all units into a single middle part. Distribute available specificity and subtask units so each later part adds new task-relevant information.
- The final part must not be a paraphrased request to fix the issue. It should add remaining concrete information: an exact artifact, contrast, example, edge case, constraint, diagnostic, target, or negative requirement when such information exists in the original prompt.
- The concatenation of all parts must contain all information from the original prompt and no outside information.
- Preservation applies to all distinct source information, not only information you consider important or necessary to solve the task. Semantically duplicate source wording may be consolidated, but every unique detail within metadata, diagnostics, environment details, examples, and overlapping statements must remain.
- Preserve exact technical artifacts from the original prompt across the parts. Copy every fenced block, exact output, stack trace, error message, line reference, and log completely and verbatim. A list of representative lines or a prose description of an artifact is not preservation.
- Tokens listed in verbatim_placeholders represent complete original artifacts that will be restored programmatically. Include every placeholder exactly once, unchanged and on its own line in the part where that pasted artifact logically belongs. Do not wrap placeholders in fences or replace them with summaries.
- `[[VERBATIM_INTERFACE_SECTION]]` represents the complete `## Interface` section. Put this placeholder on its own in a later message, normally the final part.
- If problem_statement has a nontrivial `## Interface` section, keep that heading and its complete specification together as a distinct later message, normally the final part. Exact interface names and signatures are an API contract for the tests, not optional implementation advice.
- The boilerplate `No new interfaces are introduced.` may be attached to a later message but does not by itself provide incremental value.
- Do not add labels such as "Part 1" or "Message 2".
- Keep the conversational framing natural, as if a user is gradually providing task context to an agent. It is natural for a user to paste a complete code block, stack trace, command output, or log in a later message.
- Do not preserve the original paragraph order if that creates an illogical distribution of units. Recombine the units into natural requests.
- For bugs, prefer this progression: vague symptom and affected functionality with at most one compact observed/expected clue; then a failing or suspicious concrete example; then contrasting passing examples, exact outputs, diagnostics, or final constraints.
- For feature requests, prefer this progression when the source describes one feature: broad desired capability; then concrete examples and expected UX/API; then requirements, implementation constraints, edge cases, HOW TO, and HOW NOT. When the source requests multiple subtasks, introduce them in a logical implementation sequence.
- If previous_issues is non-empty, previous_parts contains the split from the failed attempt with verbatim placeholders still unexpanded. Use the issues to revise that split rather than regenerating blindly. Return a complete corrected split, not only the changed parts.

Return JSON with keys in this exact order: specificity_units, subtask_units, parts, rationale, and coverage_check. Produce the two unit lists before parts so the completed Stage 1 decomposition is available while composing Stage 2.
In rationale, explain which units are introduced by each part and whether each later part adds specificity, a subtask, or both.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""


def build_check_prompt(
    problem_statement: str,
    parts: list[str],
    incremental_units: list[str] | None = None,
) -> str:
    payload = {
        "problem_statement": problem_statement,
        "parts": parts,
        "incremental_units": incremental_units or [],
        "required_json_schema": {
            "ok": "boolean",
            "issues": ["string"],
        },
    }
    return f"""Validate whether parts are a faithful interactive split of problem_statement.

Return ok=true only if:
- Independently reassess problem_statement using the same eligibility standard as the initial filter. It must contain at least two distinct source-grounded requirements, constraints, behaviors, edge cases, or separately implementable changes that independently affect acceptance criteria.
- Do not trust incremental_units as proof of eligibility. Treat them as claims to verify against problem_statement. If they are paraphrases of one behavioral change, inferred rather than explicit, or not independently meaningful, return ok=false.
- A reproduction, stack trace, command output, log, version, environment, screenshot, diagnostic, expected-vs-actual pair, or multiple examples of the same behavior do not create additional units of work.
- A restated request, narrower restatement of the same goal, uncertainty, intended-behavior question, ease-of-fix assessment, or closing remark does not create an additional unit of work.
- Different input variants, container types, classes, methods, or execution paths do not create separate units when they only apply the same requested behavior.
- A nontrivial `## Interface` section is valid incremental information when it adds exact public names, signatures, return types, or locations beyond the behavioral requirements. Verify that it is kept complete and together in a distinct later part. The boilerplate `No new interfaces are introduced.` does not count as a unit.
- parts[0] expresses the main idea and core goal of problem_statement and contains enough target and context for a skilled coding agent to complete a meaningful, coherent initial implementation or fix.
- parts[0] is not required to contain every acceptance criterion, requirement, interface, constraint, edge case, or subtask needed to pass the final tests.
- parts[0] may include one compact concrete symptom or expected-vs-actual contrast if needed.
- The split is logical rather than paragraph-based.
- Every later part adds substantial source-grounded information that was not disclosed earlier: increased specificity, an additional requested subtask, or both.
- Every part is non-empty and useful.
- Every part contains substantive task information. No part is only uncertainty, politeness, frustration, meta-commentary, or a closing remark.
- Every valid source-grounded item in incremental_units remains represented, but it may appear in a later part when it is a requirement, interface, constraint, edge case, or separately implementable subtask.
- Requirements and subtasks are distributed progressively instead of being forced into parts[0].
- Semantically equivalent source statements are consolidated while retaining all unique details. The parts do not repeat concrete acceptance criteria merely because the source states the same meaning more than once.
- No prose fact is presented both as a paraphrase or summary and again in its original wording.
- Each detailed requirement, constraint, example, diagnostic, and subtask appears in only one part unless repetition occurs inside a verbatim artifact that must be copied exactly.
- Allow parts[0] to state a broad core goal that overlaps with a later verbatim Interface when the Interface adds exact API names, signatures, types, return values, or locations. Do not reject this high-level-to-contract progression as duplication.
- No single middle part contains nearly all technical detail while the final part adds only a restated request.
- The final part adds new task-relevant information that was not already present in earlier parts, unless the original prompt truly contains no remaining substantive information.
- For short/simple issues, remaining concrete examples or exact outputs are sufficient final-part value. Do not reject them just because they are examples rather than a new category of information.
- The parts contain no facts, advice, paths, line numbers, tests, stack frames, or implementation details that are absent from problem_statement.
- Taken together, the parts preserve every distinct piece of information from problem_statement, whether or not it seems important for solving the task. Semantically duplicate source wording may be consolidated. If any unique detail, example, metadata, diagnostic line, qualifier, or constraint was omitted or generalized, return ok=false and identify the missing information in issues.
- Exact technical artifacts from problem_statement, such as code snippets, expected or actual outputs, stack traces, command output, logs, error messages, constraints, identifiers, and negative requirements, appear completely and verbatim somewhere in the parts. Representative samples, selected frames or lines, and prose summaries do not satisfy this requirement.
- Natural conversational phrasing and distributed specificity must not cause compression. Complete pasted technical artifacts are natural user-message content.

Return JSON with keys ok and issues.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""
