#!/usr/bin/env bash
source ./.venv/bin/activate

RND=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)

python -m emulator.emulator_manual_client \
    --keys-json-path "./.cache/keys_${RND}.json" \
    --mp4-path "./.cache/output_${RND}.mp4" \
    --manual-mode \
    --fps 60
