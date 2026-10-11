# QA proof phase prompt

Delivered after the answer is frozen. The agent writes and submits an executable `qa_proof/v1` proof. `{{...}}` fields are filled in by the harness; `{{task}}` is the prompt text from the dataset record.

````text
<prompt_kind>repository_question_proof</prompt_kind>
<question_id>{{question_id}}</question_id>

Proof phase for the answer the harness has already frozen. The repository stays
read-only for this whole phase.

Original question:
<question>
{{task}}
</question>

Exact immutable answer:
<frozen_answer>
{{frozen_answer}}
</frozen_answer>

Write the executable Bash proof script only to this harness-selected path:
<proof_path>{{proof_path}}</proof_path>

Every non-empty stdout line must be one JSON object with protocol exactly
qa_proof/v1, a non-empty observation or observation_id string, a non-empty
method string, and a top-level evidence or data field containing the observed
payload. Do not place observed payload fields only at the top level.

Apply every evidence, isolation, and clean-state requirement from the system
message to this script. In particular: report evidence and method instead of an
established, correct, passed, or verdict boolean; use no hidden-answer
comparison, hardcoded verdict, circular evidence, or symbol-existence-only
check; establish every essential edge of a behavioral, causal, invariant, or
performance claim through executable behavior, runtime instrumentation, a
controlled intervention, or a controlled counterfactual; include a
counterfactual or negative observation where practical; run active positive and
adversarial counterexamples for a universal claim; use REPO_ROOT and assert the
exact expected source files under it; fail in a no-repository counterfactual
instead of falling back to installed packages or build caches; keep temporary
files outside REPO_ROOT with cleanup and restoration traps; and leave the
worktree and index unchanged.

The script is re-executed later against a checkout that may not contain your
repository changes, so prove the answer from pre-existing code only. It must
produce the same observations on the repository as it stood before any edit of
yours and on its current state, and must not import, read, execute, or assert the
existence of anything you added, moved, renamed, or altered.

Writing, making executable, running, and checking the proof must happen before
submission. Never put COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, submission
commands, this proof path or its directory, or commands that modify the proof
file inside the proof script. Do not repeat or replace the answer, and do not
print the sentinel in this phase.

When the fixed executable script is ready, call submit_proof with an empty
object as the only tool call. The harness retrieves the proof file directly,
runs a preflight execution, and returns exact feedback unless the script exits
zero and every stdout observation follows the required schema.
````
