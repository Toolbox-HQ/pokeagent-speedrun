. .venv/bin/activate
python -m models.inference.build_merged_embeddings --input_folder .cache/pokeagent/dinov2  --output_folder .cache/pokeagent/db_embeddings --splits 10