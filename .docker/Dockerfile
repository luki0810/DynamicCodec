# Public-friendly Dockerfile for DynamicCodec.
#
# Use this when you cannot pull mirrors.tencent.com/dct/facodec_lukilu:v1.2.
# Builds an image with all runtime dependencies the project needs.
#
# Build:
#   docker build -t dynamiccodec:local -f .docker/Dockerfile .docker/
#   # (or let scripts/setup_container.sh do it for you)
#
# Run:
#   scripts/setup_container.sh --image dynamiccodec:local
#
# We deliberately do NOT COPY the project code. Code is bind-mounted at run time
# (-v $REPO_ROOT:/app), so this image stays small and reusable across branches.

# Public PyTorch image on Docker Hub. CUDA 12.4 is the most recent CUDA tag
# PyTorch publishes for 2.6.0; if you need CUDA 11.8 instead, pin
# `pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel`.
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for audio I/O.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        sox \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Python deps. Pinned where the project depends on a specific behaviour;
# loose otherwise so future torch/cuda combos don't break us.
RUN pip install --upgrade pip && \
    pip install \
        descript-audiotools==0.7.2 \
        argbind==0.3.9 \
        vocos==0.1.0 \
        typeguard \
        humanfriendly \
        einops \
        soundfile \
        librosa==0.10.1 \
        pyworld \
        accelerate \
        tensorboard

ENV PYTHONPATH=/app

CMD ["bash"]
