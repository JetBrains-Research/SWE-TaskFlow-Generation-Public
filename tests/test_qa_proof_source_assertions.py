"""Which source-assertion idioms count as proving a file exists under REPO_ROOT.

The preflight originally recognised only two spellings: an inline
``test -f "$REPO_ROOT/lib/x.py"`` and an array of bare relative paths looped over
with ``test -f "$REPO_ROOT/$var"``. Sol happened to write one of those every time,
so nobody noticed the others were rejected. Terra spreads its assertions over four
spellings, and 222 of 731 issues in the first full Terra run died at ingestion
with "must explicitly assert at least one exact source file under REPO_ROOT"
despite every single proof asserting its sources.

The accepting tests below pin the idioms. The rejecting tests pin the property
that must survive: the script has to actually test a concrete literal path that
is known to sit under REPO_ROOT.
"""

from __future__ import annotations

import pytest

from interactive_benchmark_generation.qa_io import _validate_golden_proof_source

HEADER = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    'REPO_ROOT="${REPO_ROOT:?required}"\n'
    "PROOF_REQUIRED_TOOLS=(bash)\n"
    "command -v bash >/dev/null || { printf 'missing required proof tool\\n' >&2; exit 2; }\n"
)


def check(body: str) -> list[str]:
    return _validate_golden_proof_source(HEADER + body, "q-001.sh")["asserted_source_files"]


def test_inline_path_is_accepted():
    assert check('[[ -f "$REPO_ROOT/src/package.py" ]]\n') == ["src/package.py"]


def test_scalar_holding_a_rooted_path_is_accepted():
    body = 'SOURCE_FILE="$REPO_ROOT/src/package.py"\n[[ -f "$SOURCE_FILE" ]] || exit 2\n'
    assert check(body) == ["src/package.py"]
    braced = 'SOURCE_FILE="${REPO_ROOT}/src/other.py"\ntest -f "${SOURCE_FILE}"\n'
    assert check(braced) == ["src/other.py"]


def test_literal_word_list_loop_is_accepted():
    body = (
        "for source_file in src/package.py src/runtime.py; do\n"
        '    [[ -f "$REPO_ROOT/$source_file" ]] || exit 2\n'
        "done\n"
    )
    assert check(body) == ["src/package.py", "src/runtime.py"]


def test_backslash_continued_word_list_is_accepted():
    body = (
        "for source_file in \\\n"
        "    src/package.py \\\n"
        "    src/runtime.py; do\n"
        '    if [ ! -f "$REPO_ROOT/$source_file" ]; then exit 2; fi\n'
        "done\n"
    )
    assert check(body) == ["src/package.py", "src/runtime.py"]


def test_array_of_rooted_paths_tested_bare_is_accepted():
    body = (
        'EXPECTED=("$REPO_ROOT/src/package.py" "$REPO_ROOT/src/runtime.py")\n'
        'for source_path in "${EXPECTED[@]}"; do\n'
        '    [[ -f "$source_path" ]] || exit 2\n'
        "done\n"
    )
    assert check(body) == ["src/package.py", "src/runtime.py"]


def test_relative_paths_tested_bare_are_accepted_only_after_cd_to_repo_root():
    loop = (
        'EXPECTED=("src/package.py")\n'
        'for source_path in "${EXPECTED[@]}"; do\n'
        '    [[ -f "$source_path" ]] || exit 2\n'
        "done\n"
    )
    assert check('cd "$REPO_ROOT"\n' + loop) == ["src/package.py"]

    # No cd at all: a bare relative test proves nothing about REPO_ROOT.
    with pytest.raises(ValueError, match="exact source file under REPO_ROOT"):
        check(loop)

    # cd'd away since: the working directory is no longer known to be REPO_ROOT.
    with pytest.raises(ValueError, match="exact source file under REPO_ROOT"):
        check('cd "$REPO_ROOT"\ncd /tmp\n' + loop)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            'SOURCE_FILE="$REPO_ROOT/src/package.py"\nprintf "%s" "$SOURCE_FILE"\n',
            "assigning a path without testing it",
        ),
        (
            'SOURCE_FILE="/etc/passwd"\n[[ -f "$SOURCE_FILE" ]]\n',
            "a path outside the repository",
        ),
        (
            'SOURCE_FILE="$REPO_ROOT/$discovered"\n[[ -f "$SOURCE_FILE" ]]\n',
            "an interpolated, non-literal path",
        ),
        (
            'for f in "${DISCOVERED[@]}"; do [[ -f "$REPO_ROOT/$f" ]]; done\n',
            "an array that was never declared literally",
        ),
        (
            "for f in src/package.py; do printf '%s' \"$f\"; done\n",
            "a loop that never tests the file",
        ),
        (
            '[[ -f "$REPO_ROOT/.git/HEAD" ]]\n',
            "a git metadata file rather than a source file",
        ),
        (
            'for f in ../outside.py; do [[ -f "$REPO_ROOT/$f" ]]; done\n',
            "a path escaping the repository",
        ),
    ],
)
def test_rejects_assertions_that_do_not_prove_a_repository_source(body, reason):
    with pytest.raises(ValueError, match="exact source file under REPO_ROOT"):
        check(body)
    assert reason


def test_python_must_still_assert_sources_before_running():
    late = (
        'cd "$REPO_ROOT"\n'
        'export PYTHONPATH="$REPO_ROOT/src"\n'
        "export PYTHONNOUSERSITE=1\nexport PYTHONSAFEPATH=1\nexport PYTHONDONTWRITEBYTECODE=1\n"
        "python3 -c 'print(1)'\n"
        'SOURCE_FILE="$REPO_ROOT/src/package.py"\n[[ -f "$SOURCE_FILE" ]]\n'
    )
    with pytest.raises(ValueError, match="assert repository source before running Python"):
        _validate_golden_proof_source(HEADER + late, "q-001.sh")
