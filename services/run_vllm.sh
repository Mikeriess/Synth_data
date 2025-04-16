python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-27b-it \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000
