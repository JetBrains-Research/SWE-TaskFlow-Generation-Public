# SWE-TaskFlow: qutebrowser search URL example

This document follows one concrete task through the stages named in the
[SWE-TaskFlow pipeline diagram](images/taskflow-benchmark-pipeline.svg).
Its QA-first record is
[row 52 of the `qa_first_terra` split](https://huggingface.co/datasets/JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro/viewer/default/qa_first_terra?row=52).

Instance:
`instance_qutebrowser__qutebrowser-fec187c2cb53d769c2682b35ca77858a811414a8-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d`

The final section describes the observed QA-first Luna run rather than an ideal
or synthetic outcome. In that run, the QA answer and proof passed, while the
issue patch failed one required test.

## How to read this walkthrough

`LLM` marks a model call and is followed by links to the exact prompts.

`DETERMINISTIC` marks code or verifier execution without a model call.

`ARTIFACT` marks a durable output that becomes another stage's input.

The complete path is:

```text
SOURCE ISSUE
  ├─ 0A: eligibility → unit extraction → prompt composition → validation
  │       └─ ARTIFACT: Prompt 1 → Prompt 2 → Prompt 3
  └─ 0B: QA generation → golden-proof execution → adversarial review
          └─ ARTIFACT: q-2 + canonical answer + golden proof

STAGE 1: join by instance ID
  ├─ visible to Luna: q-2 → Prompt 1 → Prompt 2 → Prompt 3
  └─ hidden for scoring: canonical answer + proofs + executions

STAGE 2: Luna answers q-2, proves it, then attempts the issue
  ├─ ARTIFACT: frozen answer + agent proof
  └─ ARTIFACT: cumulative source patch

STAGE 3
  ├─ issue verifier → FAIL
  └─ proof execution → Sol judge → PASS
```

## 0 · Build reusable inputs for this qutebrowser task · two independent lanes

The benchmark preparation begins with two independent transformations:

- [0A](#stage-0a) turns the original issue into three ordered issue-solving
  prompts.
- [0B](#stage-0b) turns the repository and a previous solving trajectory into a
  validated, proof-carrying repository question.

The two outputs are reusable independently. They are joined only in
[Stage 1](#stage-1).

<a id="stage-0a"></a>

### 0A · SPLIT THE QUTEBROWSER ISSUE INTO ORDERED PROMPTS

<a id="stage-0a-source"></a>

#### 0A. SOURCE ISSUE

The source task is the
[original SWE-bench Pro record](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/viewer/default/test?row=590).
Its generated TaskFlow record is also available in the
[published multi-prompt dataset](https://huggingface.co/datasets/JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro/viewer/default/split_prompt?row=679).

Its core request is to fix search URL construction in qutebrowser:

- Search terms must be URL-encoded when inserted into query parameters.
- Spaces must become `%20` or another appropriate URL-safe representation.
- Terms containing characters such as hyphens and spaces must be handled
  consistently.
- The behavior must remain correct across different host domains.
- No new public interface is introduced.

The source issue deliberately does not prescribe the implementation. It states
the externally visible behavior that the agent must recover and implement.

<a id="stage-0a-eligibility"></a>

#### 0A. CHECK SPLIT ELIGIBILITY

`LLM`

**Exact prompts:** [system](../src/interactive_benchmark_generation/prompts.py#L21-L23)
· [classification](../src/interactive_benchmark_generation/prompts.py#L26-L72)

> **PROMPT CONTRACT**
>
> Accept the issue only if it contains at least two independent,
> source-grounded requirement units. Do not count reproductions, diagnostics,
> repeated examples, restatements, or interface boilerplate as additional
> units. Back every accepted unit with an exact source quote.

**Observed transformation**

For this task, Luna classified the issue as `incremental_requirements` and
identified two independently testable units:

1. Properly URL-encode search terms when constructing query parameters.
2. Preserve correct parameter encoding across different host domains.

The recorded filter reason says that these are separate acceptance conditions,
so the task is eligible for progressive disclosure.

> **OUTPUT →** Accepted source issue passed to
> [PASS 1 · EXTRACT UNITS](#stage-0a-extract).

<a id="stage-0a-extract"></a>

#### 0A. PASS 1 · EXTRACT UNITS

`LLM` · first pass of the generation call

**Exact prompts:** [system](../src/interactive_benchmark_generation/prompts.py#L3-L13)
· [unit extraction and split](../src/interactive_benchmark_generation/prompts.py#L75-L159)

> **PROMPT CONTRACT**
>
> Extract specificity units and subtask units from the source. Consolidate
> duplicate wording, but preserve every unique constraint, identifier,
> qualifier, example, and verbatim technical artifact. Do not invent
> requirements merely to make the task easier to split.

**Observed transformation**

For this issue, the generation record reports six specificity units and one
subtask unit. Together they cover:

- The affected search URL construction behavior.
- The existing encoding failure.
- URL encoding when building query parameters.
- Space encoding as `%20` or another URL-safe form.
- Consistent handling of hyphens and spaces.
- Correct behavior across different host domains.
- The unchanged interface contract.

> **OUTPUT →** Source-grounded units passed directly to
> [PASS 2 · WRITE ORDERED PROMPTS](#stage-0a-compose).

<a id="stage-0a-compose"></a>

#### 0A. PASS 2 · WRITE ORDERED PROMPTS

`LLM` · second pass of the same generation call

**Exact prompts:** [system](../src/interactive_benchmark_generation/prompts.py#L3-L13)
· [split composition](../src/interactive_benchmark_generation/prompts.py#L75-L159)

> **PROMPT CONTRACT**
>
> Compose exactly the requested number of progressive user messages. Make the
> first message a meaningful initial implementation task. Make every later
> message add source-grounded information that was not available earlier.
> Preserve all source details across the complete sequence, without unsupported
> additions or unnecessary repetition. Keep technical artifacts verbatim.

**Observed transformation**

The resulting three prompts are:

##### Prompt 1

> The URL utility functions need a fix for constructing search URLs from search
> terms that contain special characters or spaces. The current search URL
> construction may not properly encode special characters.

This prompt establishes the affected subsystem, the core bug, and a meaningful
first change without exposing every acceptance criterion.

##### Prompt 2

> Update the search URL construction function so search terms are properly
> URL-encoded when building query parameters. Spaces should be encoded as `%20`
> or another appropriate URL-safe format, and terms containing special
> characters such as hyphens and spaces must be handled consistently.

This continuation adds the exact query-parameter requirement and concrete
encoding constraints.

##### Prompt 3

> The construction must also work correctly with different host domains while
> preserving proper parameter encoding. Follow the complete interface contract
> below.
>
> ## Interface
>
> No new public interfaces are introduced.

This continuation adds the cross-host requirement and reveals the preserved
interface contract.

The recorded rationale confirms that the source units are distributed across
the prompts exactly in that progression: core symptom first, concrete encoding
behavior second, and cross-host plus interface constraints last.

> **OUTPUT →** Candidate sequence passed to
> [FINAL MODEL VALIDATION](#stage-0a-validation).

<a id="stage-0a-validation"></a>

#### 0A. FINAL MODEL VALIDATION

`LLM` · independent validator

**Exact prompts:** [system](../src/interactive_benchmark_generation/prompts.py#L16-L18)
· [validation](../src/interactive_benchmark_generation/prompts.py#L162-L210)

> **PROMPT CONTRACT**
>
> Independently recheck split eligibility, prompt progression, source coverage,
> verbatim-artifact preservation, semantic duplication, and unsupported
> additions. Do not trust the generator's extracted units or rationale. Return
> an `ok` decision together with structured issues.

**Observed validation**

For this task:

- Status: `accepted`.
- Number of parts: `3`.
- Model: `openai/gpt-5.6-luna`.
- Validation issues: none.
- Coverage check: all six specificity units and the one subtask unit are
  assigned exactly once.

Retries would have reused the same split prompt with the previous candidate and
validation feedback. No retry-specific prompt is used.

<a id="stage-0a-output"></a>

#### 0A. ORDERED ISSUE-SOLVING PROMPTS

`ARTIFACT`

The output is the exact ordered sequence `Prompt 1 → Prompt 2 → Prompt 3`.
Collectively, the sequence reconstructs the original issue without adding a new
requirement.

> **OUTPUT → INPUT** This artifact becomes the
> [Stage 1 ISSUE-PROMPT DATASET input](#stage-1-issue-input).

<a id="stage-0b"></a>

### 0B · BUILD A PROOF-CARRYING QA REFERENCE FROM THE SAME TASK

<a id="stage-0b-input"></a>

#### 0B. INPUTS

The QA lane receives:

- The original qutebrowser task and repository checkout.
- A previous successful issue-solving trajectory produced by Luna.

The retained trajectory contains 30 messages, uses `openai/gpt-5.6-luna`, ends with
`Submitted`, and includes a submitted patch. The QA generator uses both this
record of repository exploration and its own direct inspection of the original
repository.

<a id="stage-0b-generation"></a>

#### 0B. EXTRACT FACTS AND FORM QUESTIONS

`LLM`

**Exact prompts:** [QA bundle generation](../prompts/qa/generation.md)
· [golden-proof skeleton](../src/interactive_benchmark_generation/templates/golden_proof_skeleton.sh)

> **PROMPT CONTRACT**
>
> Inspect both the repository and the issue-solving trajectory. Generate the
> requested number of non-overlapping, multi-hop repository questions. For each
> candidate, provide a repository fact, question, canonical answer, source
> evidence hops, reasoning chain, and executable Bash golden proof.
>
> Keep every question and proof valid both before and after the issue fix.
> Establish behavioral claims with executable runtime evidence rather than
> static source matching. Follow the supplied proof skeleton for output
> protocol, repository isolation, tool checks, temporary files, and clean-state
> handling.

**Observed transformation**

For the released QA-first example, one patch-stable question is retained:

##### Selected fact `f2`

Search-engine input resolves a configured first-token shortcut to that engine.
If the first token is unknown, it remains part of the search term and the
required `DEFAULT` engine is used. The resulting term is percent-quoted before
the selected URL template is expanded.

##### Question `q-2`

> How does configured search-engine input distinguish a recognized shortcut
> from an unknown leading word, and how is the resulting term carried into the
> selected search URL?

##### Canonical answer

The configuration requires a `DEFAULT` template and may define named templates
with a `{}` placeholder. Input is stripped and split once. A leading token is
treated as a shortcut only when it is a configured key; otherwise, the entire
input remains the term and `DEFAULT` is selected. The term is percent-quoted
with no safe characters, inserted into the selected template, and converted
through `qurl_from_user_input`.

##### Evidence chain

The question requires four connected repository locations:

1. `qutebrowser/config/configdata.yml` defines the required `DEFAULT` key and
   named search-engine templates.
2. `_parse_search_term` recognizes only configured leading tokens and preserves
   an unknown leading token as part of the full term.
3. `_get_search_url` selects `DEFAULT` when no engine was recognized and quotes
   the term before template expansion.
4. `qurl_from_user_input` converts the formatted string into a `QUrl`.

> **OUTPUT →** `q-2`, its reference answer, evidence chain, and golden proof
> passed to [EXECUTE EACH PROOF](#stage-0b-proof-execution).

<a id="stage-0b-proof-execution"></a>

#### 0B. EXECUTE EACH PROOF

`DETERMINISTIC` · Harbor execution · no LLM prompt

The benchmark verifier executes the generated golden proof in a clean task
environment. It checks the process result, evidence protocol, repository
dependency, determinism, and repository-state integrity.

For `q-2`, the proof executes two controlled configurations:

- Recognized shortcut:
  `named alpha/beta & gamma` is parsed as engine `named` and term
  `alpha/beta & gamma`, producing
  `https://named.invalid/find?term=alpha%2Fbeta%20%26%20gamma&source=qute`.
- Unknown shortcut:
  `unknown alpha/beta` is parsed with no named engine and the complete term is
  retained, producing
  `https://default.invalid/?term=unknown%20alpha%2Fbeta`.

Execution result:

- Exit code: `0`.
- Evidence valid: `true`.
- Repository dependent: `true`.
- Deterministic: `true`.
- Repository state unchanged: `true`.

> **OUTPUT →** Candidate plus trusted execution report passed to
> [REVIEW EACH CANDIDATE](#stage-0b-review).

<a id="stage-0b-review"></a>

#### 0B. REVIEW EACH CANDIDATE

`LLM` · adversarial reviewer

**Exact prompt:** [proof-aware adversarial review](../prompts/qa/review.md)

> **PROMPT CONTRACT**
>
> Try to defeat each candidate rather than merely confirming it. Check
> repository grounding, multi-hop necessity, reasoning validity, answer
> ambiguity, category fit, and proof quality. Treat trusted execution results as
> authoritative. Do not accept a behavioral claim primarily on AST, regex, or
> source-text inspection; require the proof to exercise the relevant behavior.

**Observed review**

The review accepts `q-2` because:

- The question requires both parsing and URL-construction hops.
- Its wording covers recognized and unknown leading-token alternatives.
- The proof runs the repository implementation under controlled configuration.
- It demonstrates term retention, `DEFAULT` fallback, percent quoting, and
  template expansion.
- It leaves the repository unchanged.

The conditional
[golden-proof repair prompt](../prompts/qa/golden-proof-repair.md)
would repair only a rejected proof script while freezing the question and
canonical answer. This candidate did not need that repair path.

<a id="stage-0b-output"></a>

#### 0B. VALIDATED QA REFERENCE SET

`ARTIFACT`

The retained output for this task contains:

- Question `q-2`.
- Fact `f2`.
- The canonical answer.
- Four evidence hops and a three-step reasoning chain.
- The executable golden proof.
- The trusted golden-proof execution report.
- The patch-stable selection marking `q-2` as usable before or after the issue
  fix.

These references are held back from the inference agent.

> **OUTPUT → INPUT** This artifact becomes the
> [Stage 1 QA REFERENCE DATASET input](#stage-1-qa-input).

<a id="stage-1"></a>

## 1 · Assemble the qutebrowser QA-first TaskFlow sequence

`DETERMINISTIC` · record assembly · no LLM prompt

<a id="stage-1-issue-input"></a>

#### 1. ISSUE-PROMPT DATASET

> **INPUT ←** [0A. ORDERED ISSUE-SOLVING PROMPTS](#stage-0a-output)

```text
Prompt 1 → Prompt 2 → Prompt 3
```

<a id="stage-1-qa-input"></a>

#### 1. QA REFERENCE DATASET

> **INPUT ←** [0B. VALIDATED QA REFERENCE SET](#stage-0b-output)

```text
q-2 + canonical answer + golden proof + trusted execution
```

<a id="stage-1-combine"></a>

#### 1. COMBINE BY TASK ID

The builder joins the two records by the full qutebrowser instance ID and
produces a typed `qa_multistep_prompt/v1` record.

For the QA-first configuration, the visible prompt order is:

1. QA question `q-2`.
2. Issue Prompt 1.
3. Issue Prompt 2.
4. Issue Prompt 3.

The record also contains hashes binding it to the source multi-prompt record,
the QA bundle, and the original source task.

<a id="stage-1-inference-sequence"></a>

#### 1. PROMPT SEQUENCE SENT TO INFERENCE AGENT

Only the question and issue-prompt text is exposed:

```text
q-2 → Prompt 1 → Prompt 2 → Prompt 3
```

The agent does not receive the canonical answer, golden proof, evidence chain,
review, or trusted execution report.

> **OUTPUT → INPUT** This visible sequence becomes the
> [Stage 2 INFERENCE AGENT INPUT](#stage-2-input).

<a id="stage-1-validation-inputs"></a>

#### 1. VALIDATION INPUTS

The following data is stored separately for post-inference scoring:

- The canonical answer.
- The golden proof.
- The trusted golden-proof execution.
- The clean original task.

> **OUTPUT → INPUT** These hidden artifacts become inputs to
> [Stage 3 QA proof execution](#stage-3-proof-execution) and
> [Stage 3 semantic judging](#stage-3-judge).

<a id="stage-2"></a>

## 2 · Run inference in Harbor

Harbor starts an isolated qutebrowser task environment and runs
`openai/gpt-5.6-luna` with medium reasoning effort.

The per-phase agent prompts are in [`prompts/agent/`](../prompts/agent/).

<a id="stage-2-input"></a>

#### 2. INFERENCE AGENT INPUT

> **INPUT ←**
> [Stage 1 PROMPT SEQUENCE SENT TO INFERENCE AGENT](#stage-1-inference-sequence)

```text
q-2 → Prompt 1 → Prompt 2 → Prompt 3
```

The agent receives:

- A clean qutebrowser checkout under `/app`.
- One persistent conversation across all four typed prompts.
- The prompt-kind-specific answer, proof, and issue contracts.
- No validation references.

The shared system prompt tells the agent to obey the current
`prompt_kind`, submit that phase before the next prompt can arrive, preserve
cross-phase boundaries, and keep QA artifacts out of source patches.

<a id="stage-2-qa"></a>

#### 2. IF THIS IS A QA PROMPT

This graph block contains two consecutive LLM phases.

##### QA answer phase

`LLM`

**Exact prompt:** [`qa_answer_template`](../prompts/agent/qa-answer.md)

> **PROMPT CONTRACT**
>
> Inspect the repository without modifying it and answer only about pre-existing
> behavior. Keep the answer valid both before and after the issue fix. Write it
> to the harness-selected path outside the repository and freeze it verbatim.
> Submit only the direct answer: no proof, reasoning, Markdown, or preface.

**Observed answer**

Luna submitted:

> The input is stripped and split into at most two whitespace-separated parts.
> With two parts, the first is a recognized shortcut only if it is a key in
> `url.searchengines`; then the remainder is the term and that engine is
> selected. An unknown first word is not discarded: the whole original input is
> the term and `DEFAULT` is selected. The term is URL-quoted with no safe
> characters and inserted into the selected engine URL template by replacing
> its `{}` placeholder.

Answer phase result:

- Status: submitted.
- Steps: `4`.
- Tool calls: `6`.
- Latency: `11.18 s`.
- Cost: `$0.0028368`.

##### QA proof phase

`LLM`

**Exact prompt:** [`proof_template`](../prompts/agent/qa-proof.md)

> **PROMPT CONTRACT**
>
> Treat the quoted answer as frozen and keep the repository read-only. Write one
> executable Bash proof implementing `qa_proof/v1`, run it, check it, and submit
> it through the dedicated proof tool. Emit structured observations with a
> method and evidence. Establish behavioral claims through runtime execution or
> a controlled counterfactual. Resolve imports from `REPO_ROOT`, keep temporary
> files outside the repository, and leave the worktree and index unchanged.

**Observed proof**

The submitted proof executes:

- `ddg alpha beta`, confirming configured shortcut selection and `%20` encoding.
- `mystery alpha beta`, confirming that the unknown leading word remains in the
  term and `DEFAULT` is used.
- `ddg a/b?c d`, confirming percent encoding of `/`, `?`, and spaces.
- Whitespace-only input, confirming the pre-existing `ValueError` behavior.

Proof phase result:

- Preflight: passed.
- Status: submitted.
- Steps: `17`.
- Tool calls: `18`.
- Latency: `160.04 s`.
- Cost: `$0.01409126`.

<a id="stage-2-issue"></a>

#### 2. IF THIS IS AN ISSUE PROMPT

`LLM`

**Exact prompt:** [`issue_template`](../prompts/agent/issue.md)

> **PROMPT CONTRACT**
>
> For the current issue message, modify regular source files but never tests or
> configuration. Implement only what the task context revealed so far
> justifies; do not assume later constraints. End every prompt with a separate
> patch checkpoint. If a continuation requires no new change, submit an
> explicitly empty checkpoint.

**Observed issue work**

For Prompt 1, Luna changed one line in
`qutebrowser/utils/urlutils.py`:

```diff
-    quoted_term = urllib.parse.quote(term, safe='')
+    quoted_term = QUrl.toPercentEncoding(term).data().decode('ascii')
```

Prompt 1 produced a `608 B` patch checkpoint. Prompt 2 and Prompt 3 produced
empty checkpoints because Luna considered the first change sufficient for the
later constraints. The cumulative patch therefore remained `608 B`.

Issue phase result:

- Three submitted issue records.
- Total issue steps: `13`.
- Issue latency: `48.11 s`.
- Issue cost: `$0.02022494`.

<a id="stage-2-shared-output"></a>

#### 2. ISSUE / SHARED OUTPUTS

`ARTIFACT`

The run produces:

- Three issue patch checkpoints.
- A final cumulative `608 B` patch.
- The complete QA and issue conversation trajectory.
- A run manifest recording all four typed prompt submissions.

> **OUTPUT → INPUT** The cumulative patch becomes
> [Stage 3 FINAL CUMULATIVE PATCH](#stage-3-final-patch).

<a id="stage-2-qa-output"></a>

#### 2. QA AGENT OUTPUTS

`ARTIFACT`

The QA phases produce:

- The frozen answer for `q-2`.
- The executable agent proof.
- The preflight observations.

> **OUTPUT → INPUT** These artifacts become
> [Stage 3 QA SUBMISSIONS](#stage-3-qa-submissions).

<a id="stage-3"></a>

## 3 · Score this qutebrowser run on two independent paths

The two branches consume outputs from the same inference run but answer
different questions. The benchmark does not combine them into one scalar.

<a id="stage-3-issue"></a>

### 3A · ISSUE SCORING · SAME PATCHED HARBOR TASK SANDBOX

<a id="stage-3-final-patch"></a>

#### 3A. FINAL CUMULATIVE PATCH

`ARTIFACT`

> **INPUT ←** [Stage 2 ISSUE / SHARED OUTPUTS](#stage-2-shared-output)

- One source file changed.
- One encoding expression replaced.
- Cumulative patch size: `608 B`.

QA answers and proofs are not part of this patch.

<a id="stage-3-issue-validation"></a>

#### 3A. ISSUE PATCH VALIDATION

`DETERMINISTIC` · task test execution · no LLM prompt

The task verifier runs `tests/unit/utils/test_urlutils.py` in the same patched
Harbor sandbox used for inference.

The relevant failure is:

```text
test_get_search_url[test path-search-www.qutebrowser.org-q=path-search-True]
expected host: www.qutebrowser.org
actual host:   www.example.org
```

With `url.open_base_url = True`, the input `test path-search` must open the base
URL configured under the term `path-search`. Luna's patch changes percent
encoding but does not implement that base-URL selection behavior, so the
required case still fails.

<a id="stage-3-issue-result"></a>

#### 3A. ISSUE RESULT

- Collected: `244`.
- Passed: `242`.
- Failed: `1`.
- Skipped: `1`.
- Harbor SWE reward: `0`.

The issue-resolution branch therefore fails even though the QA branch below
passes.

<a id="stage-3-qa"></a>

### 3B · QA SCORING · SEPARATE CLEAN POST-PROCESSING TASKS

<a id="stage-3-qa-submissions"></a>

#### 3B. QA SUBMISSIONS

`ARTIFACT`

> **INPUT ←** [Stage 2 QA AGENT OUTPUTS](#stage-2-qa-output)

- The frozen `q-2` answer.
- The submitted `q-2` proof script.

> **HIDDEN INPUT ←** [Stage 1 VALIDATION INPUTS](#stage-1-validation-inputs)

- The canonical answer and golden proof.
- The trusted golden execution.
- A clean copy of the original qutebrowser repository.

<a id="stage-3-proof-execution"></a>

#### 3B. EXECUTE AGENT PROOFS

`DETERMINISTIC` · separate clean Harbor task · no LLM prompt

The proof is re-executed in a separate clean task, without Luna's issue patch.
This checks that the answer and proof describe pre-existing behavior rather than
behavior introduced during issue resolution.

Agent-proof execution result:

- Submitted: `true`.
- Executed: `true`.
- Success: `true`.
- Exit code: `0`.
- Timed out: `false`.
- Output truncated: `false`.
- Repository state unchanged: `true`.
- Duration: `0.81 s`.

The observations reproduce configured-shortcut selection, unknown-token
fallback, reserved-character encoding, and whitespace-only rejection.

> **OUTPUT →** Trusted agent-proof execution passed to
> [JUDGE ANSWER + PROOF](#stage-3-judge).

<a id="stage-3-judge"></a>

#### 3B. JUDGE ANSWER + PROOF

`LLM` · `openai/gpt-5.6-sol`

**Exact prompt:** [proof-carrying QA judge](../prompts/qa/judge.md)

> **PROMPT CONTRACT**
>
> Judge the question using the canonical answer, golden proof, frozen agent
> answer, submitted agent proof, and trusted executions. Treat golden artifacts
> as references rather than exhaustive oracles, and accept valid alternatives.
> Treat proof output as observations, never as a verdict. Require runtime
> evidence for behavioral and causal claims. Decide separately whether the
> answer is correct, whether the proof matches it, and whether the proof
> establishes it.

**Observed judgment**

For `q-2`, the judge reports:

- Agent answer correct: `yes`.
- Agent proof matches answer: `yes`.
- Agent proof establishes answer: `yes`.
- Golden proof supports golden answer: `yes`.
- Gold reference status: `sufficient`.
- Relation: `same_as_golden`.
- Confidence: `0.99`.
- Dataset issue: `false`.

The rationale states that the proof directly executes the essential behavioral
edges against the repository and independently reproduces shortcut selection,
unknown-token retention, `DEFAULT` fallback, and percent encoding.

<a id="stage-3-qa-result"></a>

#### 3B. QA RESULT

- Answer Accuracy: `100%`.
- Answer + Proof Accuracy: `100%`.
- Proof submission rate: `100%`.
- Proof execution rate: `100%`.
- Proof establishment rate: `100%`.
- Dataset issue rate: `0%`.

The complete observed outcome is therefore:

> **OBSERVED FINAL RESULT**
>
> - Repository understanding: **PASS**
> - Executable proof: **PASS**
> - Issue resolution: **FAIL**

