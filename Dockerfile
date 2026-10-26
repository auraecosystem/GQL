# gpu/Dockerfile.gql
# GQL — General GPU Layer
# NVIDIA CUDA runtime foundation for LMLM workloads.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GQL_HOME=/opt/gql \
    MODEL_DIR=/models \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system gql \
    && useradd --system --gid gql \
       --home-dir /opt/gql --create-home gql \
    && mkdir -p /opt/gql /models /cache/huggingface \
    && chown -R gql:gql /opt/gql /models /cache

WORKDIR /opt/gql

# Install pinned dependencies from the build context.
# Provide gpu/requirements-gpu.txt before building.
COPY requirements-gpu.txt /tmp/requirements-gpu.txt

RUN python3 -m pip install --no-cache-dir \
      --requirement /tmp/requirements-gpu.txt \
    && rm /tmp/requirements-gpu.txt

COPY --chown=gql:gql . /opt/gql

USER gql

VOLUME ["/models", "/cache"]

# The application must implement this module before serving works.
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "lmlm"]
