#!/usr/bin/env bash
set -e

# Variables
IMAGE_NAME="pokeagent"
BUILD_DIR=".cache/containers"
TAG="latest"

docker build -f dconfig/Dockerfile -t "$IMAGE_NAME:$TAG" .
docker save "$IMAGE_NAME:$TAG" -o "$BUILD_DIR/${IMAGE_NAME}_${TAG}.tar"
echo "Docker image built!"
