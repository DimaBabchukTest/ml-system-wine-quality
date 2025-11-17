FROM debian:bookworm-slim

# 1) Install Python + basic build tools + curl (for uv install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2) Install uv (Astral) as a binary
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Put uv on PATH for login shells and non-login shells
ENV PATH="/root/.local/bin:${PATH}"

# 3) Set working directory
WORKDIR /code

# 4) Copy project metadata first (for better layer caching)
COPY pyproject.toml uv.lock ./

# 5) Create and sync venv using uv (locked)
ENV UV_PROJECT_ENVIRONMENT=.venv
RUN uv sync --locked
RUN uv lock --check

# 6) Copy the actual application code and model 
COPY ./app /code/app
COPY ./model_artifact /code/model_artifact

# 7) DEBUG: prove they are there at build time
RUN echo "==== AFTER COPY ====" && ls -R /code

ENV PYTHONPATH=/code

EXPOSE 8000

# 8) Run FastAPI app via fastapi CLI
# Assuming your app object is in app/main.py
CMD ["/code/.venv/bin/python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


### docker build -t wine_model_docker .
### docker run -p 127.0.0.1:8000:8000 wine_model_docker
### docker build --no-cache -t wine_model_docker .
### docker run -it --entrypoint /bin/bash wine_model_docker