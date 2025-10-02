#!/usr/bin/env bash
set -e

# Variables
IMAGE_NAME="pokeagent"
BUILD_DIR="./docker_build"
TAG="latest"

# Build Docker image
docker build -t "$IMAGE_NAME:$TAG" .

# Save Docker image to build directory
#docker save "$IMAGE_NAME:$TAG" -o "$BUILD_DIR/${IMAGE_NAME}_${TAG}.tar"

echo "Docker image built!"
