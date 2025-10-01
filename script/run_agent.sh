source ./.venv/bin/activate

RND=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)
STATE_NUM=$((RANDOM % 10))

python main.py \
    --max-steps 100000 \
    --save-s3 "test_filter" \
    --policy "random_policy" \
    --save-state "./state/${STATE_NUM}.state" \
    --keys-json-path "./Data/keys_${RND}.json" \
    --mp4-path "./Data/output_${RND}.mp4"

rm -r ./Data
