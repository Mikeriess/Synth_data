python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-9b-it \
    --gpu-memory-utilization 1.0 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000