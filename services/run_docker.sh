docker run -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:23.12-py3 bash