WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) ./.venv/bin/torchrun --nnodes=1 --nproc_per_node=1 ./models/inference/agent_inference.py --config ./config/online/online_debug.yaml
