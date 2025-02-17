python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.3-70B-Instruct \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 \
    --port 8000
