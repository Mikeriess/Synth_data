# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the inspection service files
COPY services/inspect /app/

# Copy the dataset_processor module (since it's imported from parent directory)
COPY services/inspect/dataset_processor.py /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p datasets/default

# Expose the port the server runs on
EXPOSE 8000

# Make the server accessible on all network interfaces
ENV HOST=0.0.0.0

# Run the server
CMD ["python", "server.py"] 