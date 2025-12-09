#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)
export WANDB_DEBUG=true
wandb sync .cache/pokeagent/tmp/wandb/offline-run*

