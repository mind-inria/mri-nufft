#!/usr/bin/env bash
# Fail on public API breaking changes since the latest release that are not
# acknowledged in .github/api-breaks.txt.
#
# Usage: .github/check_api.sh [REF]   (REF defaults to the latest release tag)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ref=${1:-$(git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude '*-*')}
acknowledged_file=.github/api-breaks.txt
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export LC_ALL=C

echo "Checking the public API against $ref"
status=0
uvx griffe==2.4.1 check mrinufft -s src -a "$ref" -f oneline --no-color \
    >"$tmp/griffe.txt" 2>&1 || status=$?

# Keep the reported breakages (griffe prints them on stderr), without their line
# numbers, which move with unrelated edits.
grep -E '^[^ :]+\.py:[0-9]+: ' "$tmp/griffe.txt" |
    sed -E 's/^([^:]+):[0-9]+: /\1: /' | sort -u >"$tmp/found.txt" || true
if [ "$status" -ne 0 ] && [ ! -s "$tmp/found.txt" ]; then
    cat "$tmp/griffe.txt" >&2
    echo "griffe failed (exit code $status)." >&2
    exit "$status"
fi
grep -vE '^[[:space:]]*(#|$)' "$acknowledged_file" | sort -u >"$tmp/acknowledged.txt" || true

comm -13 "$tmp/found.txt" "$tmp/acknowledged.txt" | while read -r line; do
    echo "::warning file=$acknowledged_file::Acknowledged but no longer reported: $line"
done

comm -23 "$tmp/found.txt" "$tmp/acknowledged.txt" >"$tmp/new.txt"
if [ -s "$tmp/new.txt" ]; then
    echo "::error file=$acknowledged_file::Breaking changes since $ref. Go through a deprecation (mrinufft._deprecation), or acknowledge them by adding these lines to $acknowledged_file:"
    cat "$tmp/new.txt"
    exit 1
fi
echo "No unacknowledged breaking change since $ref."
