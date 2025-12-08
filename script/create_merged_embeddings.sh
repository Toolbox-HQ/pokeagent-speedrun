. .venv/bin/activate
python -m models.inference.build_merged_embeddings --folder .cache/pokeagent/dinov2  --out-prefix .cache/pokeagent/db_embeddings --splits 10