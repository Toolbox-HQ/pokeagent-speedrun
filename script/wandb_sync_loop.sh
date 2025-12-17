#!/bin/bash
source .venv/bin/activate

while true; do
  wandb beta sync .cache/pokeagent/tmp/wandb/offline-run*
  sleep 1800  # 30 minutes
done