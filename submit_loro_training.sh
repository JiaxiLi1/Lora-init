#!/bin/bash
#SBATCH --job-name=loro_training          # Job name
#SBATCH --partition=gpu                   # Partition name (adjust as needed)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --mem=32G                        # Memory per node
#SBATCH --time=24:00:00                  # Wall time limit (hh:mm:ss)
#SBATCH --output=loro_training_%j.out    # Standard output and error log
#SBATCH --error=loro_training_%j.err     # Error log

# Load singularity/apptainer module if needed
# module load singularity

# Set the Docker image name (replace with your actual image name)
DOCKER_IMAGE="loro_sparse:latest"

# Set the container bind mounts (adjust paths as needed)
# Bind current directory to /workspace in container
BIND_MOUNTS="-B $(pwd):/workspace"

# Optional: Bind additional data directories if needed
# BIND_MOUNTS="$BIND_MOUNTS -B /path/to/data:/data"

echo "Starting LORO training job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Change to the working directory
cd /workspace

# Run the training command inside the singularity container
singularity exec --nv $BIND_MOUNTS docker://$DOCKER_IMAGE \
    bash -c "source activate loro_sparse && \
    torchrun --nproc_per_node 1 --master_port 29504 run_c4.py \
    --model_config configs/llama_130m.json \
    --dtype bfloat16 \
    --batch_size 1 \
    --total_batch_size 1 \
    --num_training_steps 50 \
    --save_every 2000 \
    --eval_every 10 \
    --lr 0.01 \
    --scheduler cosine_restart \
    --warmup_steps 5 \
    --min_lr_ratio 0.1 \
    --cosine_restart_freq 500 \
    --lr_adjust_steps -2000 \
    --weight_decay 0 \
    --optimizer loro_adamw \
    --loro_refresh all \
    --loro_refresh_freq 10 \
    --loro_scope all \
    --loro_init xavier \
    --loro_attn_rank 64 \
    --loro_mlp_rank 64 \
    --loro_type loro \
    --loro_freq 10 \
    --loro_lr_scaler -1 \
    --c4_local False \
    --enable_2to4_sparse \
    --sparse_init_scale 1.0"

echo "Training job completed at $(date)"

# Optional: Copy important files back to a persistent location
# cp -r checkpoints /path/to/persistent/storage/
# cp -r logs /path/to/persistent/storage/ 