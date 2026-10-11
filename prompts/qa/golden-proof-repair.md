# Golden-proof repair prompt

Sent only when strict validation rejects a golden proof. `golden-repair-request.json` lists the affected questions and the exact validation failures.

````text
Repair only the golden proof scripts listed in /qa/input/bundle/golden-repair-request.json.
The question and golden_answer are immutable. Do not rewrite, omit, merge, or replace any question. Do not change the claimed answer.
For each requested question, inspect /app and replace the current script with a strict executable proof of the same answer. Address every exact validation failure in the request.
Behavioral, causal, invariant, and performance claims require executable runtime evidence, controlled intervention, instrumentation, or a controlled counterfactual. Static source extraction cannot be the primary evidence for those claims.
Every script must use REPO_ROOT, assert exact expected source files, isolate repository imports from global packages, declare and check required tools, use temporary files outside the repository, disable bytecode writes, and leave the worktree and index unchanged.
Follow this skeleton exactly; the structural checks accept these spellings, and a proof that improvises around them is rejected before it ever runs.
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
Every non-empty stdout line must be one JSON object with protocol qa_proof/v1, a non-empty observation or observation_id, a non-empty method, and evidence or data. Diagnostics belong on stderr.
Do not include agent submission sentinels, git staging, git commits, or cached patch application in a proof script.
Write exactly /qa/output/repairs.json with protocol qa_golden_proof_repairs/v1 and repairs. repairs must contain exactly one object for each requested question_id, with only question_id and script. script must be the complete replacement Bash source.
The file must have exactly this shape:
```json
{"protocol": "qa_golden_proof_repairs/v1",
 "repairs": [{"question_id": "q-001", "script": "#!/usr/bin/env bash ..."}]}
```
````
