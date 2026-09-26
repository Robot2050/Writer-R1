export WANDB_PROJECT="Qwen3-Writer"
export WANDB_API_KEY=""

CUDA_VISIBLE_DEVICES=3,4,5,6 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model sft_Qwen3-4B-writer \
    --model_type qwen3_nothinking \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8121 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset merged_RL_shuffled.jsonl \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --truncation_strategy left \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --max_completion_length 4096 \
    --max_length 8192 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --num_generations 8 \
    --dynamic_sample true \
    --max_resample_times 3 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --temperature 1 \
    --top_p 0.9 \
    --top_k 50 \
    --deepspeed zero3 \
    --num_iterations 1 \
    --beta 0.01 \
    --num_train_epochs 2 \
    --report_to wandb \
    --external_plugins reward.py \
    --reward_funcs writer_reward \
    --eval_strategy steps \
    --eval_steps 20 \
    --logging_steps 1 \
    --log_completions true \
    --log_entropy true \
    --save_steps 20 \
    --save_total_limit 3 \
    --save_only_model true \
    --output_dir gspo_Qwen3-4B \
    --run_name gspo_Qwen3-4B


# nohup bash gspo_qwen3.sh > gspo_Qwen3-4B.log 2>&1 &

