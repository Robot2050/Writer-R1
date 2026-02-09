
export WANDB_PROJECT="writer_Qwen3-4B"
export WANDB_NAME="gspo_Qwen3-4B"
export wandb_exp_name="gspo_Qwen3-4B"

CUDA_VISIBLE_DEVICES=1,2 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model train/records_all_19K_qwen3_4B \
    --model_type qwen3_nothinking \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8121 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset train_bash/records_RL_05K_shuffled.jsonl \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --truncation_strategy left \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --max_completion_length 4096 \
    --max_length 1536 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --num_generations 8 \
    --dynamic_sample true \
    --max_resample_times 3 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
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
    --external_plugins train_bash/reward.py \
    --reward_funcs writer_reward \
    --eval_strategy steps \
    --eval_steps 20 \
    --logging_steps 1 \
    --log_completions true \
    --log_entropy true \
    --save_steps 20 \
    --save_total_limit 5 \
    --save_only_model true \
    --output_dir train_bash/result/gspo_Qwen3-4B

# nohup bash gspo_qwen3.sh > result/gspo_qwen3-4B.log 2>&1 &


