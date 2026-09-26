
CUDA_VISIBLE_DEVICES=1,2 \
swift rollout \
    --model sft_Qwen3-4B-writer \
    --model_type qwen3_nothinking \
    --vllm_tensor_parallel_size 2 \
    --host 127.0.0.1 \
    --port 8121 \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 13000 \
    --vllm_max_num_seqs 8 \
    --vllm_enable_prefix_caching true \
    --template qwen3_nothinking \


# nohup bash vllm_qwen3.sh > log/vllm_qwen3.log 2>&1 &