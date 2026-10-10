#!/bin/bash
#SBATCH --job-name=bev_eval_full
#SBATCH --output=bev_eval_full_%j.log
#SBATCH --partition=sooner_gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00

source /scratch/kaustubh/miniconda3/bin/activate bev_env
cd /scratch/kaustubh/bevformer_project/BEVFormer

export PYTHONPATH=$PYTHONPATH:/scratch/kaustubh/bevformer_project/mmdetection3d:.

python -u run_full_eval.py
