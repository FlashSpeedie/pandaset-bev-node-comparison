#!/bin/bash
#SBATCH --job-name=bev_base
#SBATCH --output=bev_base_%j.log
#SBATCH --partition=sooner_gpu_test
#SBATCH --container=el9hw
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

export PATH="/scratch/kaustubh/miniconda3/envs/bev_env/bin:$PATH"
export CONDA_PREFIX="/scratch/kaustubh/miniconda3/envs/bev_env"
export PYTHONPATH="/scratch/kaustubh/bevformer_project/mmdetection3d:.:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd /scratch/kaustubh/bevformer_project/BEVFormer

# Launch test directly without distributed wrapper
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    --eval bbox \
    --out results_base_3d_bbox.pkl \
    --cfg-options data.workers_per_gpu=0
