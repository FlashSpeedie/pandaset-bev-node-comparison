#!/bin/bash
#SBATCH --job-name=bev_infer
#SBATCH --output=bev_infer_%j.log
#SBATCH --partition=sooner_gpu_test
#SBATCH --container=el9hw
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

# Set environment paths and python dependencies
export PATH="/scratch/kaustubh/miniconda3/envs/bev_env/bin:$PATH"
export CONDA_PREFIX="/scratch/kaustubh/miniconda3/envs/bev_env"
export PYTHONPATH="/scratch/kaustubh/bevformer_project/mmdetection3d:.:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd /scratch/kaustubh/bevformer_project/BEVFormer

# Launch dist_test.sh
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_small.py \
    ckpts/bevformer_small_epoch_24.pth \
    1 \
    --eval bbox \
    --out results_3d_bbox.pkl \
    --cfg-options data.workers_per_gpu=0
