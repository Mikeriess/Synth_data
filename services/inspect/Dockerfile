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

# Set Hugging Face token as environment variable
ENV HF_TOKEN=${HUGGING_FACE_HUB_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Create symbolic links for HTML files in the working directory
CMD cd services/inspect && \
    for file in *.html; do \
        ln -sf /workspace/services/inspect/$file /workspace/$file; \
    done && \
    ln -sf /workspace/services/inspect/inspector.js /workspace/inspector.js && \
    cd /workspace && \
    python services/inspect/server.py 