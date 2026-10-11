# QA bundle generation prompt

Sent to the generator agent inside the SWE-bench Pro task container. The repository is mounted at `/app`, the issue-solving trajectory at `/qa/input/trajectory.json`, and outputs are collected from `/qa/output`. `<instance_id>` is the SWE-bench Pro instance ID and `<n_questions>` is the requested number of questions (3 in the released dataset).

````text
Generate the QA benchmark artifacts for the repository in /app.
Read the existing issue-solving trajectory from /qa/input/trajectory.json.
Use both the trajectory and direct repository observations.
Use this exact instance_id everywhere: <instance_id>.
Use schema_version "2.0" and evaluation_mode proof_judged_v1.
Write /qa/output/bundle.json containing schema_version, instance_id, documentation, facts, questions, metadata, debug, and golden_proofs. Also write questions.jsonl and executable golden proof scripts under /qa/output/golden-proofs.
Each fact must contain exactly fact_id and fact. Each question must contain fact_id, question_id, question, category, golden_answer, golden_proof, evidence_hops, reasoning_chain, and evaluation_mode. golden_answer must contain canonical.
golden_proof must be one flat object with exactly path, format, sha256, timeout_seconds, and max_output_bytes. format must be bash. The SHA-256 must match the exact UTF-8 script bytes. Raw generation must omit golden_proof_validation.
golden_proofs must map every golden_proof.path to the exact script source. Write the same source to its separate relative file.
golden_answer and golden_proof are references for a semantic judge, not exhaustive answer or proof oracles. Do not generate incorrect answers, closed answer contracts, or answer-comparison probes.
Every fact and accepted answer must describe the current repository state in /app. Do not ask about desired behavior or code that exists only in the trajectory patch. A respondent must be able to recover every answer by inspecting /app without seeing the issue, trajectory, or documentation.
The same question and the same golden proof must hold both before and after the issue is fixed, because both are asked again on a repository that an agent has already patched. The gold patch is not in this task, but /qa/input/trajectory.json shows which files, symbols and behaviour the fix changes. Treat everything the trajectory changes as unstable ground: do not make an answer component depend on a value, signature, branch or message that the fix alters, adds or removes, and do not write a proof whose observations would change once the fix is applied. Prefer facts about surrounding mechanism that the fix relies on but does not modify. Choose expected source files and assertions that exist in both states. When a candidate cannot be made stable this way, reject it and record the reason in debug rather than weakening the question.
Every golden proof must be an executable Bash script. It must use REPO_ROOT instead of assuming /app and emit bounded JSON-line observations using protocol qa_proof/v1.
Inspect the repository and available tools before choosing a proof method. After inspection, name exact expected source files and assert that each exists under REPO_ROOT before using it. Resolve project imports only from REPO_ROOT; disable global import fallbacks and fail with a diagnostic when a required repository module or tool is unavailable.
Every script must declare its executable dependencies in one literal line using PROOF_REQUIRED_TOOLS=(tool1 tool2). Check every declared tool with command -v and emit a diagnostic containing 'missing required proof tool' before exiting when unavailable. For Python, set PYTHONPATH with REPO_ROOT first, set PYTHONNOUSERSITE=1, PYTHONSAFEPATH=1, and PYTHONDONTWRITEBYTECODE=1, change directory to REPO_ROOT, and assert the expected source files before the first Python command.
Follow this skeleton exactly; the structural checks accept these spellings, and a proof that improvises around them is rejected before it ever runs. Replace the source files, tools and observation with the real ones, and keep every line that is not a placeholder.
```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:?REPO_ROOT must name the repository root}"

PROOF_REQUIRED_TOOLS=(python3)
for tool in "${PROOF_REQUIRED_TOOLS[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || {
        printf 'missing required proof tool: %s\n' "$tool" >&2
        exit 2
    }
done

EXPECTED_SOURCES=(pkg/module_a.py pkg/module_b.py)
for source_file in "${EXPECTED_SOURCES[@]}"; do
    [[ -f "$REPO_ROOT/$source_file" ]] || {
        printf 'required repository source unavailable: %s\n' "$source_file" >&2
        exit 2
    }
done

export PYTHONPATH="$REPO_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO_ROOT"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python3 - <<'PYTHON'
import json
print(json.dumps({
    "protocol": "qa_proof/v1",
    "observation_id": "observed-behaviour",
    "observation": "what running the repository code showed",
    "method": "how it was obtained, e.g. executed pkg.module_a with a crafted input",
    "evidence": {"input": "...", "output": "..."},
}))
PYTHON
```
Each observation must report repository-derived evidence and its method, not a verdict. Scripts must not emit or rely on an established, correct, passed, or equivalent boolean. The verifier and judge determine validity.
Reject symbol-existence-only checks, hardcoded verdicts, embedded golden answer comparisons, circular evidence, and scripts that merely print source literals without deriving the claimed behavior. Require an executable counterfactual or negative observation where practical.
For behavioral, causal, invariant, and performance claims, every essential causal edge must be established by executable behavior, runtime instrumentation, a controlled intervention, or a controlled counterfactual. AST parsing, regex matching, and source inspection may support structural facts, but they must not be the primary evidence for behavior or causality. Reject a candidate when its essential chain depends mainly on static source extraction, unless source structure is itself the exact claim.
Proof scripts must finish without network access, avoid persistent repository changes, exit zero only after collecting supporting observations, fail when required evidence is unavailable, and keep total output below their max_output_bytes.
Set PYTHONDONTWRITEBYTECODE=1 for Python commands. Do not write into the repository, create bytecode, install tools, or silently substitute a different diagnostic when the inspected tool is unavailable.
Use only these category values: what_architecture_exploration, what_concept_definition, what_dependency_tracing, why_design_rationale, why_purpose_exploration, why_performance, where_data_control_flow, how_system_design, how_algorithm_implementation, and how_api_framework_support.
Do not generate feature-location or identifier-location questions. Reject any candidate answerable by one identifier lookup, one search, or one isolated snippet. Questions asking which property, class, method, function, enum member, identifier, or expression is used are forbidden.
Generate questions that require reconstruction of a rationale, dependency chain, control or data flow, architectural mechanism, invariant, counterfactual consequence, failure mode, or non-trivial algorithm. Every answer component must require a distinct evidence hop. Reject a candidate if one function, method, class, file, test, docstring, or isolated snippet directly contains the complete answer, even when extra unnecessary evidence can be cited. At least one answer component must depend on a cross-file transition. Do not combine independent triggers with and/or into one question. Choose a category only after establishing the exact repository flow. Do not force category diversity within one issue.
Use why_design_rationale only when repository comments, tests, history, or architecture explicitly support the rationale rather than merely the observed behavior. Use why_performance only for an evidenced performance tradeoff. A where_data_control_flow question must ask the respondent to reconstruct a cross-component path, not explain a local implementation.
Use these only as style seeds: why a request passes through normalization before caching; how state crosses parser, model, and rendering layers; why two similar execution paths intentionally diverge; how a validation guard prevents a downstream invariant violation; and how a value is transformed across multiple repository components.
Privately draft at least 2 × <n_questions> candidate flow scopes. Run every check below before selecting the strongest <n_questions>. Do not pad the output with a weaker local question. For every draft, name an entry symbol and a different terminal symbol, verify both symbols, and trace the exact cross-file edges between them. Reject the draft if one function, one adjacent helper, or one file contains the complete answer. Reject corroborative tests or citations that are not necessary to derive an answer field. Search for a second repository-grounded path that would also answer the wording; if one exists, narrow the question to the named terminal symbol or reject the draft. Record rejected drafts and reasons in debug.
Every evidence_hops list must contain at least three observations across at least two repository files. Each hop must contain evidence_id, file, symbol, and claim, and every hop must be necessary for the golden answer. Every reasoning_chain must contain at least three ordered steps with step_id, claim, and evidence_ids. Each reasoning claim must state the concrete repository-specific transition it proves. Generic claims such as an input reaching a component, logic transforming an input, or an outcome following are forbidden. Every edge must cite the evidence that directly establishes that edge, including intermediate helper calls and final decision points. Store the complete evidence and reasoning chain but do not reveal them in the question.
Questions must request concise phase-one answers without exposing proof instructions, golden answers, evidence, reasoning chains, or private data. Do not add interpretations such as intent, stability, safety, malformed input, or guaranteed outcomes unless the cited repository evidence explicitly establishes them. Account for fallback and empty-value paths in the golden answer and proof.
Generate exactly <n_questions> non-overlapping candidate questions.
Every golden proof path must match golden-proofs/<instance_id>/q-*.sh.
You may run each golden proof as a generator self-check and record that self-check in debug. It is not trusted golden validation. Trusted golden validation runs later in a separate Harbor verifier stage.
Do not read or use hidden gold patches, source verifier tests, or pre-existing probes because they are not present in this task.
````
