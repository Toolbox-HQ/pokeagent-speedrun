#!/usr/bin/env bash
source ./.venv/bin/activate

RND=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)

python -m emulator.emulator_manual_client \
    --keys-json-path "./.cache/agent_eval/${RND}.json" \
    --mp4-path "./.cache/agent_eval/${RND}.mp4" \
    --manual-mode \
    --fps 60 \
    --save-state $1 \
    --rom .cache/rom.gba