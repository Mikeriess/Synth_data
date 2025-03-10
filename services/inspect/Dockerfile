# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory to the inspect directory
WORKDIR /app

# Install dependencies
COPY services/inspect/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p datasets/default

# Expose the port the server runs on
EXPOSE 8000

# Make the server accessible on all network interfaces
ENV HOST=0.0.0.0

# Set Hugging Face token as environment variable
ENV HF_TOKEN=${HUGGING_FACE_HUB_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Run the server
CMD ["python", "server.py"] 