source ./.venv/bin/activate

RND=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)

python -m emulator.emulator_client \
    --keys-json-path "./emulator/data/keys_${RND}.json" \
    --mp4-path "./emulator/data/output_${RND}.mp4" \
    --manual-mode \
    --fps 60