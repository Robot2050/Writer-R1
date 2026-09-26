#!/bin/bash

# 设置 wandb 为离线模式
# export WANDB_MODE=offline
# export WANDB_PROJECT=

# 设置 wandb 为在线模式
export WANDB_PROJECT="Qwen3-Writer"
export WANDB_API_KEY=""


CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29801 FORCE_TORCHRUN=1 llamafactory-cli train train_bash1/qwen3_full_sft.yaml

# nohup bash train_bash1/full_sft_qwen3.sh > train_bash1/sft_Qwen3-4B-writer.log 2>&1 &
