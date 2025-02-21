python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/llama-3.1-405b-instruct-fp8 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 \
    --port 8000
