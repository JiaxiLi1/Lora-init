#!/bin/bash

# LORO + 2:4 Sparse Docker Build Script (Minimal)
# ===============================================
# Based on existing working 2by4-pretrain-acc-examples/Dockerfile
# Only adds LORO-specific dependencies

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🐳 Building LORO + 2:4 Sparse Training Docker Environment (Minimal)"
echo "=================================================================="
echo "Project root: $PROJECT_ROOT"
echo "Based on: 2by4-pretrain-acc-examples/Dockerfile + LORO additions"

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
    echo "❌ Error: sparse package not found"
    echo "   Please build the 2by4-pretrain-acc-examples sparse package first"
    exit 1
fi

echo "✅ Project structure validated"

# Set Docker image name and tag
IMAGE_NAME="loro-2by4-minimal"
IMAGE_TAG="latest" 
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "🏗️  Building Docker image: $FULL_IMAGE_NAME"
echo "   Using minimal approach: 2by4 base + LORO packages"
echo "   This should be faster than full rebuild..."

# Build the Docker image
cd "$PROJECT_ROOT"
docker build \
    -f LORO-main/Dockerfile.loro_2by4_minimal \
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
    echo "# Available commands inside container:"
    echo "  python verify_loro_2by4.py     # Verify installation"
    echo "  ./run_loro_training.sh         # Start LORO + 2:4 training"
    echo ""
    echo "📁 Container structure:"
    echo "  /workspace/2by4-pretrain-acc-examples/  # Original 2by4 project"
    echo "  /workspace/LORO-main/                   # LORO framework" 
    echo "  /workspace/verify_loro_2by4.py          # Verification script"
    echo "  /workspace/run_loro_training.sh         # Training script"
    echo ""
    echo "💡 Key differences from original:"
    echo "  ✓ Environment name: loro_2by4 (instead of 2by4)"
    echo "  ✓ Added LORO packages: transformers 4.46.3, accelerate, loguru, etc."
    echo "  ✓ Both projects available in container"
    echo "  ✓ Ready for LORO + 2:4 sparse training"
    echo ""
else
    echo "❌ Docker build failed!"
    exit 1
fi 