#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)

# Directory containing wandb runs (default is ./wandb)
WANDB_DIR=${1:-"./wandb"}

echo "Starting periodic wandb sync for directory: $WANDB_DIR"
echo "Sync interval: 4 minutes"

while true; do
    echo "[$(date)] Running wandb sync..."
    # Sync all offline runs (recursively finds *.wandb or offline dirs)
    find "$WANDB_DIR" -type d -name "offline-run-*" | while read -r run; do
        echo "Syncing: $run"
        wandb sync "$run" --include-offline
    done
    echo "[$(date)] Sync complete. Sleeping..."
    sleep 240  # 4 minutes
done
