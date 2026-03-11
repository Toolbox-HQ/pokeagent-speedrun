#!/bin/bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:?Usage: $0 <destination>}"

EXCLUDES=(
    .cache
    .venv
    .triton
    wandb
    tmp
    checkpoints
)

SKIP_ONLY=(
    core.*
    slurm-*.out
)

RSYNC_ARGS=(--archive --delete --info=progress2)
for dir in "${EXCLUDES[@]}"; do
    RSYNC_ARGS+=(--exclude="$dir")
done
for pat in "${SKIP_ONLY[@]}"; do
    RSYNC_ARGS+=(--exclude="$pat")
done

rsync "${RSYNC_ARGS[@]}" "$SRC/" "$DST/"

for dir in "${EXCLUDES[@]}"; do
    src_dir="$SRC/$dir"
    dst_link="$DST/$dir"

    [ -e "$src_dir" ] || continue

    if [ -L "$dst_link" ]; then
        rm "$dst_link"
    elif [ -e "$dst_link" ]; then
        echo "Warning: $dst_link exists and is not a symlink, skipping"
        continue
    fi

    ln -s "$src_dir" "$dst_link"
    echo "Linked: $dst_link -> $src_dir"
done
