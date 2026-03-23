#!/bin/bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATE="$(date +%Y-%m-%d)"
BACKUP_ROOT="$(dirname "$REPO")/backup/$DATE"

DIRS=(
    .cache/pokeagent/tmp_checkpoints
    .cache/pokeagent/online
    .cache/pokeagent/tmp
)

for dir in "${DIRS[@]}"; do
    src="$REPO/$dir"
    dst="$BACKUP_ROOT/$(basename "$dir")"

    [ -e "$src" ] || { echo "Skipping $dir (not found)"; continue; }

    mkdir -p "$dst"
    echo "Moving $src -> $dst"
    find "$src" -maxdepth 1 -mindepth 1 -exec mv -t "$dst" {} +
    echo "Done: $(basename "$dir")"
done
