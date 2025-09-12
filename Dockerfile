FROM nvidia/cuda:12.1-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set up Python alias
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Set work directory
WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir tensorflow[and-cuda]
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the project in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/results/models /app/results/logs

# Set permissions
RUN chmod +x /app/scripts/*.py

# Expose ports for Jupyter and API services
EXPOSE 8888 8000 7860

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "jupyter" ]; then\n\
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""\n\
elif [ "$1" = "api" ]; then\n\
    python -m src.inference.model_serving\n\
elif [ "$1" = "train" ]; then\n\
    python scripts/run_experiments.py\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["jupyter"]