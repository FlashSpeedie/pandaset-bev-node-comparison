#!/bin/bash
#SBATCH --job-name=bevformer_test
#SBATCH --output=bev_output_%j.log
#SBATCH --error=bev_error_%j.log
#SBATCH --partition=sooner_gpu_test,gpu
#SBATCH --gres=gpu:1
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# 1. Activate Conda environment
source /scratch/$USER/miniconda3/bin/activate
conda activate bev_env

# 2. Log confirmation
echo "=== Running BEVFormer GPU Test on Sooner Cluster ==="
nvidia-smi

# 3. Check CUDA Visibility
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
