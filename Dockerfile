# Use NVIDIA CUDA devel image (required for CUDA compilation) - using 12.1.1 to support compute_89
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1

# Install system dependencies including compilation tools
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    vim \
    curl \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Create the loro_sparse environment with Python 3.8
RUN conda create -n loro_sparse python=3.8 -y

# Activate the environment for subsequent commands
SHELL ["/bin/bash", "--login", "-c"]

# Install PyTorch with CUDA support via pip (more reliable for specific versions)
RUN conda activate loro_sparse && \
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies (matching loro_sparse environment)
RUN conda activate loro_sparse && \
    pip install \
    numpy==1.24.4 \
    transformers==4.31.0 \
    datasets==2.20.0 \
    tokenizers==0.13.3 \
    huggingface_hub==0.28.0 \
    evaluate==0.4.2 \
    bitsandbytes==0.43.1 \
    wandb==0.17.3 \
    loguru==0.7.2 \
    tensorly==0.8.1 \
    nvitop==1.3.2 \
    lion-pytorch==0.2.2 \
    pandas==2.0.0 \
    scipy==1.10.0 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.0

# Set working directory
WORKDIR /workspace

# Copy the project files
COPY . /workspace/

# Install the loro-torch package in development mode
RUN conda activate loro_sparse && \
    cd /workspace && \
    pip install -e .

# Copy and install the sparse package if it exists
RUN conda activate loro_sparse && \
    if [ -d "/workspace/sparse_package" ]; then \
        cd /workspace/sparse_package && \
        rm -rf build/ *.egg-info/ && \
        pip install -e . ; \
    fi

# Set PYTHONPATH to include the workspace
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Make sure conda environment is activated by default
RUN echo "conda activate loro_sparse" >> ~/.bashrc

# Set the default command
CMD ["/bin/bash"] 