#!/bin/bash

# Docker image name
IMAGE_NAME="loro_sparse"
TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
    echo "Image name: ${IMAGE_NAME}:${TAG}"
    
    # Show image info
    docker images | grep ${IMAGE_NAME}
    
    echo ""
    echo "To test the image locally, run:"
    echo "docker run --gpus all -it --rm -v \$(pwd):/workspace ${IMAGE_NAME}:${TAG}"
    
    echo ""
    echo "To save the image for transfer to slurm server:"
    echo "docker save ${IMAGE_NAME}:${TAG} | gzip > ${IMAGE_NAME}_${TAG}.tar.gz"
    
else
    echo "Docker build failed!"
    exit 1
fi 