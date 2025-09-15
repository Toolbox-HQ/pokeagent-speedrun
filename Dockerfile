# Use a Python base image for convenience with uv
FROM ubuntu:24.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    sudo \
    xz-utils \
    ffmpeg \
    libpng16-16 \
    libzip4 \
    libsqlite3-0 \
    libelf1 \
    liblua5.4-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv using the official script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set a working directory inside the container
WORKDIR /app

RUN wget https://github.com/mgba-emu/mgba/releases/download/0.10.5/mGBA-0.10.5-ubuntu64-noble.tar.xz && \
    tar -xf mGBA-0.10.5-ubuntu64-noble.tar.xz && \
    rm mGBA-0.10.5-ubuntu64-noble.tar.xz
RUN dpkg -i mGBA-0.10.5-ubuntu64-noble/libmgba.deb

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync
COPY . .

# Specify the default command to run when the container starts
CMD ["/app/.venv/bin/python", "/app/main.py", "--fps", "60"]