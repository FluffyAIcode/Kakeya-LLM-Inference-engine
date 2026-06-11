#!/usr/bin/env bash
# Reviewer-aid identification header library.
# Implements docs/agent-workflow-rules.md R2.
#
# Every scripts/review_*.sh must source this file and call
# print_aid_header at startup, before any pre-flight check, so the
# user (and the reviewing agent) can immediately verify "the code
# that's about to spend GPU time is the code I think it is."
#
# Usage (in a reviewer aid script):
#
#     ROOT="$(cd "$(dirname "$0")/.." && pwd)"
#     # shellcheck disable=SC1091
#     source "$ROOT/scripts/_lib/reviewer_aid_header.sh"
#     print_aid_header "$0" "loss_type=attn_distill rank=768 steps=20000"
#
# Output (printed to stderr):
#
#     ==> review_pr_k3_f_theta_train_on_vast.sh
#         Branch:        AgentMemory/v04-pr-k3-block-c-f-theta-design-and-skeleton-8e7f
#         HEAD commit:   <SHA>  "<commit subject>"
#         Repo dir:      /workspace
#         Recipe:        loss_type=attn_distill rank=768 steps=20000
#         Started at:    2026-06-10T05:43:00Z
#
# If any of these fields cannot be derived (e.g. detached HEAD, not
# a git repo), they print as "(unavailable)" — never silently empty.
# Failure to determine branch DOES NOT abort the script — the header
# is informational; aborting would cost the user even more time.

print_aid_header() {
    local script_path="${1:-$0}"
    local recipe="${2:-(no recipe provided)}"
    local script_name
    script_name="$(basename "$script_path")"

    local repo_dir branch head_sha head_subject started
    repo_dir="$(pwd)"

    if branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"; then
        :
    else
        branch="(not a git repo)"
    fi

    if head_sha="$(git rev-parse --short HEAD 2>/dev/null)"; then
        head_subject="$(git log -1 --format=%s 2>/dev/null || echo "(unknown)")"
    else
        head_sha="(unavailable)"
        head_subject="(unavailable)"
    fi

    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    {
        echo "==> $script_name"
        echo "    Branch:        $branch"
        echo "    HEAD commit:   $head_sha  \"$head_subject\""
        echo "    Repo dir:      $repo_dir"
        echo "    Recipe:        $recipe"
        echo "    Started at:    $started"
        echo
    } >&2
}

# Helper: assert a required-branch invariant. Use this when a
# reviewer aid is ONLY valid on a specific branch (e.g. an
# experiment-specific reviewer aid). Most general aids should NOT
# assert this — they let the user run on any branch and rely on
# print_aid_header for visibility.
require_branch() {
    local expected="$1"
    local actual
    actual="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "(unavailable)")"
    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: this reviewer aid requires branch '$expected'" >&2
        echo "       Currently on: '$actual'" >&2
        echo "       Run: git checkout $expected && git pull" >&2
        return 1
    fi
}

# Helper: print a "verified by agent" tag — the agent prints this
# in its response to the user, confirming it has read the actual
# code path that will execute. Mirrors docs/agent-workflow-rules.md R3.
print_agent_verification() {
    local branch="$1"; local sha="$2"; local subject="$3"; local file="$4"
    cat >&2 <<EOF
Verified by agent:
  - Branch:    $branch
  - HEAD:      $sha  "$subject"
  - Code path: $file
EOF
}
