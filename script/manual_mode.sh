#!/usr/bin/env bash
source ./.venv/bin/activate

RND=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)

python -m emulator.emulator_manual_client \
    --keys-json-path "./.cache/game_Start.json" \
    --mp4-path "./.cache/game_Start.mp4" \
    --manual-mode \
    --fps 60 \
    --rom .cache/lz/rom/lz_rom.gba