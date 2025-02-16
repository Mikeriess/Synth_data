python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-2b-it \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 8000
