# vLLM Inference Server for Gemma-3-4B-IT

This folder contains configuration for running a vLLM inference server with Google's Gemma-3-4B-IT model using Docker.

## Prerequisites

- NVIDIA GPU with at least 24GB VRAM (RTX 4090 recommended)
- Docker and Docker Compose installed
- NVIDIA Docker runtime installed
- A valid Hugging Face access token with permissions to access the Gemma model

## Setup

1. **Set your Hugging Face token as an environment variable:**

   ```bash
   export HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here
   ```

   You can also add this to your `.bashrc` or `.zshrc` file for persistence.

2. **(Optional) Customize GPU memory utilization:**

   ```bash
   export GPU_MEMORY_UTILIZATION=0.9
   ```

   Default is 0.9 (90% of available GPU memory). Lower this value if you experience out-of-memory issues.

3. **(Optional) Adjust maximum sequence length:**
   
   The default configuration limits the maximum sequence length to 8,192 tokens to fit within the RTX 4090's memory. If you need longer sequences, you'll need to decrease the `GPU_MEMORY_UTILIZATION` or use a GPU with more VRAM. Conversely, if you only need shorter sequences, you can further reduce this value to improve performance.

## Running the Server

1. **Start the server:**

   ```bash
   docker-compose up
   ```

   Or run in detached mode:

   ```bash
   docker-compose up -d
   ```

2. **First-time run:**
   The first time you run the server, it will download the model weights from Hugging Face, which may take several minutes depending on your internet connection.

3. **Verify the server is running:**

   ```bash
   curl http://localhost:8000/v1/models
   ```

   This should return information about the loaded model.

## API Usage

The server provides an OpenAI-compatible API on port 8000. Here are examples of how to use it:

### Method 1: Using requests (Most reliable)

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/v1/chat/completions"

# Request payload
payload = {
    "model": "google/gemma-3-4b-it",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the Gemma model."}
    ],
    "temperature": 0.7,
    "max_tokens": 500
}

# Send request
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print response
print(response.json()["choices"][0]["message"]["content"])
```

### Method 2: Using OpenAI Python SDK (v1.x)

```python
from openai import OpenAI

# Create client with appropriate base URL
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not used but required by the client
)

# Generate text
response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the Gemma model."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Method 3: Using OpenAI Python SDK (v0.x legacy)

```python
import openai

# Configure the legacy client
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "not-needed"  # API key not used but required

# Generate text
response = openai.ChatCompletion.create(
    model="google/gemma-3-4b-it",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the Gemma model."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

## Stopping the Server

To stop the server:

```bash
docker-compose down
```

## Troubleshooting

- **Out of memory errors**: Try reducing the `GPU_MEMORY_UTILIZATION` environment variable (e.g., to 0.8 or 0.7).
- **Model not found**: Ensure your Hugging Face token is correctly set and has the necessary permissions.
- **API connection errors**: 
  - Check if the server is running with `curl http://localhost:8000/v1/models`
  - Use the requests-based method if you're having issues with the OpenAI client
  - Make sure you're using the correct path `/v1/chat/completions` for chat models
- **Performance issues**: The Gemma-3-4B-IT model is optimized for a single GPU. For larger models, consider using a GPU with more VRAM or a multi-GPU setup.

## Additional Information

- The vLLM server uses continuous batching for efficient inference.
- This configuration is optimized for a single NVIDIA RTX 4090 GPU.
- For more advanced configuration options, see the [vLLM documentation](https://docs.vllm.ai/). 