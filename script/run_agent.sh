source ./.venv/bin/activate

RND=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)
STATE_NUM=$((RANDOM % 10))

python main.py \
    --save-s3 "emulatorv2" \
    --policy $1 \
    --save-state "./state/${STATE_NUM}.state" \
    --keys-json-path "./Data/keys_${RND}.json" \
    --mp4-path "./Data/output_${RND}.mp4"
