# GQL GPU Fabric service image. The dedicated gpu/Dockerfile.* files remain
# available for the CPU, CUDA, and ROCm Compose profiles.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/gql/gpu \
    MODEL_DIR=/models \
    GQL_INPUT_DIR=/inputs \
    GQL_JOB_DB=/cache/gql-jobs.sqlite3 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system gql \
    && useradd --system --gid gql --home-dir /opt/gql --create-home gql \
    && mkdir -p /opt/gql /models /inputs /cache/huggingface \
    && chown -R gql:gql /opt/gql /models /inputs /cache

WORKDIR /opt/gql
COPY gpu/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt
COPY --chown=gql:gql . /opt/gql

USER gql
VOLUME ["/models", "/inputs", "/cache"]
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "uvicorn", "runtime.app:app", "--host", "0.0.0.0", "--port", "8080"]
