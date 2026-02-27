#!/bin/bash
source .venv/bin/activate
wandb beta sync --skip-synced .cache/pokeagent/tmp/wandb/offline-run-*
wandb beta sync --skip-synced ./wandb/offline-run-*