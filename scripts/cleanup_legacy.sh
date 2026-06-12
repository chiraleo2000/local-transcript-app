#!/usr/bin/env bash
# Remove legacy release binaries and duplicate docs (dry-run by default).
set -euo pipefail

DRY_RUN=1
if [[ "${1:-}" == "--apply" ]]; then
  DRY_RUN=0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

remove_path() {
  local target="$1"
  if [[ ! -e "$target" ]]; then
    return
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would remove: $target"
  else
    rm -rf "$target"
    echo "removed: $target"
  fi
}

remove_path "release/v1.2.1"
remove_path "release/v1.2.2/*.exe"
remove_path "release/v1.2.2/*.zip"
remove_path "RELEASE_NOTES_v1.2.0.md"
remove_path "RELEASE_NOTES_v1.2.1.md"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete. Re-run with --apply to delete."
fi
