# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    libssl-dev \
    libffi-dev \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create working directory
WORKDIR /app

# Copy repository files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    python setup.py develop && \
    pip install fastapi uvicorn python-multipart

# Create directories for model weights
RUN mkdir -p /app/weights/LongCat-Image && \
    mkdir -p /app/weights/LongCat-Image-Edit

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default environment variables
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    T2I_CHECKPOINT=/app/weights/LongCat-Image \
    EDIT_CHECKPOINT=/app/weights/LongCat-Image-Edit \
    USE_CPU_OFFLOAD=true \
    MAX_BATCH_SIZE=1

# Run the API server
CMD ["python", "api_server.py"]
