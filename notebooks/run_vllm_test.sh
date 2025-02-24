python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/llama-3.1-8b-instruct \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --max_model_len 4096 \
    --port 8000
