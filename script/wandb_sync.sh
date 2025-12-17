#!/bin/bash
source .venv/bin/activate
wandb beta sync .cache/pokeagent/tmp/wandb/offline-run-*
