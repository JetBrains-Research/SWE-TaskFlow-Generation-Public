# QA answer phase prompt

Delivered for each repository question. The answer is frozen verbatim before the proof phase starts. `{{...}}` fields are filled in by the harness; `{{task}}` is the prompt text from the dataset record.

````text
<prompt_kind>repository_question</prompt_kind>
<question_id>{{question_id}}</question_id>
<typed_prompt_index>{{typed_prompt_index}}</typed_prompt_index>

{{task}}

Answer phase for this question. Inspect the repository with ordinary tool calls
until you can answer the question. Do not modify repository files and do not
create files inside /app. Never print COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
while inspecting the repository or preparing the answer.

Answer about the pre-existing code only. The answer is graded against a checkout
that may not contain your changes, so it must hold both on the repository as it
stood before any edit of yours and on its current state; do not describe
behaviour you introduced or name anything you added, moved, renamed, or altered.

When the concise direct answer is ready, write only that answer, with no proof,
reasoning, Markdown, label, citations, or explanatory preface, to this
harness-chosen path outside the repository:
<answer_path>{{answer_path}}</answer_path>

Writing and optionally checking that file must happen before submission. Submit
only with a separate final bash tool call containing this exact command and
nothing else:
{{answer_submit_command}}

The harness rejects every other command that prints the sentinel. The file
content is frozen verbatim and cannot be changed during the proof phase.
````
