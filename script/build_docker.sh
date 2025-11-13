#!/usr/bin/env bash
set -e

# Variables
IMAGE_NAME="pokeagent"
BUILD_DIR="./.cache/pokeagent/containers"
TAG="latest"

docker build -f dconfig/Dockerfile -t "$IMAGE_NAME:$TAG" .

mkdir -p $BUILD_DIR
docker save "$IMAGE_NAME:$TAG" -o "$BUILD_DIR/${IMAGE_NAME}_${TAG}.tar"
echo ""$IMAGE_NAME:$TAG"  saved to $BUILD_DIR/${IMAGE_NAME}_${TAG}.tar"
