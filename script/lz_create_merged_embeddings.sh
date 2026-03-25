. .venv/bin/activate
python -m models.inference.build_merged_embeddings --input_folder .cache/lz/dinov2  --output_folder .cache/lz/db_embeddings --splits 10