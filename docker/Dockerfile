# Dockerfile for openpi development environment.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container
#
# Quick start:
#   docker compose -f docker/docker-compose.headless.yaml up --build
#   docker exec -it openpi-dev-headless bash
#
# Verify environment:
#   docker exec openpi-dev-headless /.venv/bin/python -m pytest tests/test_env_verification.py -v

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH="/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    linux-headers-generic \
    build-essential \
    clang \
    curl \
    wget \
    procps \
    lsof \
    psmisc \
    tmux \
    vim \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install dependencies (project itself installed at runtime via entrypoint)
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Patch transformers for AdaRMS / KV-cache fixes
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# Healthcheck: verify Python and key imports work
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD /.venv/bin/python -c "import openpi; import jax; import torch; print('ok')" || exit 1

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
