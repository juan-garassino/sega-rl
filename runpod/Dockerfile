# sega-rl training image (archetype B). Built by .github/workflows/build-push.yml, never locally.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg tini python3 \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 -s /bin/bash researcher \
    && mkdir -p /app /workspace && chown -R researcher:researcher /app /workspace
USER researcher
ENV PATH="/home/researcher/.local/bin:/app/.venv/bin:$PATH" UV_PYTHON_INSTALL_DIR=/home/researcher/.python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY --chown=researcher:researcher pyproject.toml uv.lock ./
# pyproject pins torch 2.14.1 from the CPU index (laptop/CI); the pod gets the same version's CUDA 12.6 wheel.
RUN uv python install 3.12 \
    && uv sync --frozen --no-install-project --no-install-package torch \
    && uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1"

COPY --chown=researcher:researcher src ./src
COPY --chown=researcher:researcher runpod ./runpod
RUN uv sync --frozen --inexact --no-install-package torch \
    && chmod +x runpod/entrypoint.sh runpod/scripts/*.sh \
    && python -c "import torch, stable_retro, sega_rl.train; assert torch.version.cuda, 'CPU torch in image'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

ENV WORKSPACE_DIR=/workspace APP_DIR=/app MODE=train
ENTRYPOINT ["/usr/bin/tini", "--", "/app/runpod/entrypoint.sh"]
