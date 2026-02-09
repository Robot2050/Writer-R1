# run in another node

CUDA_VISIBLE_DEVICES=3 \
swift rollout \
    --model train/records_all_19K_qwen3_4B \
    --model_type qwen3_nothinking \
    --vllm_tensor_parallel_size 1 \
    --host 127.0.0.1 \
    --port 8121 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 6144 \
    --vllm_max_num_seqs 4 \
    --vllm_enable_prefix_caching true \
    --template qwen3_nothinking \

# nohup bash vllm_qwen3.sh > log/vllm_qwen3_4B.log 2>&1 &