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
