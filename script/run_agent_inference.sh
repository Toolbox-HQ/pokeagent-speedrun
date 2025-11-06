source ./.venv/bin/activate

RND=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)
STATE_NUM=$((RANDOM % 10))

python main.py \
    --policy "./config/policy/debug_policy.yaml" \
    --keys-json-path "./Data/keys_${RND}.json" \
    --mp4-path "./Data/output_${RND}.mp4"
