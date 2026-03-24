#!/bin/bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")/../.." && pwd)"
DST="${1:?Usage: $0 <host>:/path}"

if [[ "$DST" != *:* ]]; then
    echo "Error: destination must be a remote path in the form <host>:/path (got: $DST)" >&2
    exit 1
fi

EXCLUDES=(
    .venv
    .triton
    wandb
    tmp
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
