# Use a Python base image for convenience with uv
FROM ubuntu:24.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv using the official script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set a working directory inside the container
WORKDIR /app


COPY pyproject.toml uv.lock .python-version ./
RUN uv sync
COPY . .

#RUN source /app/.venv/bin/activate
# Specify the default command to run when the container starts
CMD ["/app/.venv/bin/python", "/app/main.py", "--fps", "60"]
