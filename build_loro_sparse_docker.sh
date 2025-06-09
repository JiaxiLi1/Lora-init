#!/bin/bash

# LORO-Sparse Docker Build Script
# This script builds a Docker image for the LORO+2:4 sparse training environment

set -e  # Exit on any error

# Configuration
IMAGE_NAME="loro-sparse"
TAG="latest"
DOCKERFILE="Dockerfile.loro_sparse"

echo "================================"
echo "Building LORO-Sparse Docker Image"
echo "================================"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Dockerfile: ${DOCKERFILE}"
echo ""

# Check if Dockerfile exists
if [ ! -f "${DOCKERFILE}" ]; then
    echo "Error: ${DOCKERFILE} not found!"
    exit 1
fi

# Build the Docker image
echo "Starting Docker build..."
docker build -f ${DOCKERFILE} -t ${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo ""
    echo "To run the container:"
    echo "  docker run -it --gpus all -v \$(pwd):/workspace ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To run with port forwarding (for Jupyter/TensorBoard):"
    echo "  docker run -it --gpus all -p 8888:8888 -p 6006:6006 -v \$(pwd):/workspace ${IMAGE_NAME}:${TAG}"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi 