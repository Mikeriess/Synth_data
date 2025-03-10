# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory to the project root
WORKDIR /workspace

# Install dependencies
COPY services/inspect/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p services/inspect/datasets/default

# Expose the port the server runs on
EXPOSE 8000

# Make the server accessible on all network interfaces
ENV HOST=0.0.0.0

# Login to Hugging Face if token is provided
CMD if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then \
        echo "Logging in to Hugging Face Hub"; \
        python -c "from huggingface_hub import login; login('$HUGGING_FACE_HUB_TOKEN')"; \
    fi && \
    python services/inspect/server.py 