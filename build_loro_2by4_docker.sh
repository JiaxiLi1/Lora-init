#!/bin/bash

# LORO + 2:4 Sparse Docker Build Script
# =====================================
# This script builds a Docker container with complete LORO + 2:4 sparse training environment

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🐳 Building LORO + 2:4 Sparse Training Docker Environment"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"

# Validate required directories exist
echo "📋 Validating project structure..."

if [ ! -d "$PROJECT_ROOT/2by4-pretrain-acc-examples" ]; then
    echo "❌ Error: 2by4-pretrain-acc-examples directory not found at $PROJECT_ROOT/"
    echo "   Please ensure the 2by4-pretrain-acc-examples project is in the parent directory"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/LORO-main" ]; then
    echo "❌ Error: LORO-main directory not found at $PROJECT_ROOT/"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/2by4-pretrain-acc-examples/sparse_package/sparse/__init__.py" ]; then
    echo "❌ Error: sparse package not found or not properly installed"
    echo "   Please build the 2by4-pretrain-acc-examples sparse package first"
    exit 1
fi

echo "✅ Project structure validated"

# Set Docker image name and tag
IMAGE_NAME="loro-2by4-sparse"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "🏗️  Building Docker image: $FULL_IMAGE_NAME"
echo "   This may take 15-30 minutes..."

# Build the Docker image
cd "$PROJECT_ROOT"
docker build \
    -f LORO-main/Dockerfile.loro_2by4 \
    -t "$FULL_IMAGE_NAME" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker build completed successfully!"
    echo ""
    echo "🚀 Quick start commands:"
    echo "========================"
    echo ""
    echo "# Run container interactively:"
    echo "docker run --gpus all -it --rm $FULL_IMAGE_NAME"
    echo ""
    echo "# Run with volume mounting (to save results):"
    echo "docker run --gpus all -it --rm \\"
    echo "    -v \$(pwd)/outputs:/workspace/outputs \\"
    echo "    $FULL_IMAGE_NAME"
    echo ""
    echo "# Start training directly:"
    echo "docker run --gpus all -it --rm $FULL_IMAGE_NAME \\"
    echo "    /bin/bash -c 'cd /workspace && ./run_loro_sparse_training.sh'"
    echo ""
    echo "# Available commands inside container:"
    echo "  ./run_loro_sparse_training.sh    # Start LORO + 2:4 sparse training"
    echo "  python verify_installation.py     # Verify 2:4 functions work"
    echo ""
    echo "📁 Container structure:"
    echo "  /workspace/LORO-main/            # LORO framework"
    echo "  /workspace/sparse_package/       # 2:4 sparse triton functions"
    echo "  /workspace/verify_installation.py # Installation verification"
    echo "  /workspace/run_loro_sparse_training.sh # Training script"
    echo ""
else
    echo "❌ Docker build failed!"
    exit 1
fi 