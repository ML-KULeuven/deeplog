FROM ubuntu:noble

ENV PYTHONUNBUFFERED=1
WORKDIR /deeplog/

# CPU-only CI base: system deps, uv-managed venv, and Python deps. No project source copied.
COPY requirements.txt /deeplog/requirements.txt


RUN apt-get update && \
    apt-get install -y software-properties-common git curl python3-dev graphviz build-essential  && \
    add-apt-repository -y ppa:swi-prolog/stable && \
    curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh" | bash && \
    apt-get install -y swi-prolog gitlab-runner && \
    rm -rf /var/lib/apt/lists/*

# Install uv and create a project-local virtual environment.
ENV UV_INSTALL_DIR="/usr/local/bin"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s
ENV PATH="/usr/local/bin:${PATH}"
RUN uv venv /deeplog/.venv
ENV VIRTUAL_ENV=/deeplog/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu126

# Install PyTorch first via the official index URLs, then the rest of the stack.
RUN if [ "$DEVICE" = "gpu" ]; then \
      uv pip install --no-cache torch torchvision --index-url "$TORCH_CUDA_INDEX_URL"; \
    else \
      uv pip install --no-cache torch torchvision --index-url "$TORCH_CPU_INDEX_URL"; \
    fi

RUN uv pip install --no-cache -r requirements.txt
