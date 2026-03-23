#!/bin/bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$REPO/.cache/pokeagent/checkpoints"

echo "Finding optimizer.pt files in $TARGET ..."
echo ""

mapfile -t FILES < <(find "$TARGET" -name "optimizer.pt" -type f)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No optimizer.pt files found."
    exit 0
fi

for f in "${FILES[@]}"; do
    echo "  $f"
done

echo ""
echo "Total: ${#FILES[@]} files"
echo ""
read -rp "Delete all of the above? [y/N] " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

for f in "${FILES[@]}"; do
    rm "$f"
done

echo "Deleted ${#FILES[@]} files."
