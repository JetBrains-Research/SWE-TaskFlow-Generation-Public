# Issue phase prompt

Delivered for each issue-solving prompt. Every issue prompt ends with a separate patch checkpoint. `{{...}}` fields are filled in by the harness; `{{task}}` is the prompt text from the dataset record.

````text
<prompt_kind>issue_resolution</prompt_kind>
<part_index>{{part_index}}</part_index>
<typed_prompt_index>{{typed_prompt_index}}</typed_prompt_index>

{{task}}

Issue phase for this user message. Modify only regular source files in /app,
never tests or configuration files, and make only the source-code changes
justified by the task context seen so far; do not assume details that a later
user message may add.

Submit a patch for this user message in three separate tool calls:

1. Create patch.txt in one command:
   `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source
   files you modified, or `: > patch.txt` if this message needs no new
   source-code change. Do NOT commit.
2. Inspect patch.txt in a separate command and confirm it contains only your
   intended source changes with `--- a/` and `+++ b/` headers, or that it is
   correctly empty.
3. Submit with a separate final bash tool call containing this exact command and
   nothing else:
{{issue_submit_command}}

Keep tests, reproduction scripts, helper tools, build, packaging, and
configuration files, binary artifacts, and every question or proof artifact out
of patch.txt. An empty patch is a valid submission; skipping the submission is
not.
````
