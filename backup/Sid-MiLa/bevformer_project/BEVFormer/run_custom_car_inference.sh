#!/bin/bash
#SBATCH --job-name=bev_custom
#SBATCH --output=bev_custom_%j.log
#SBATCH --partition=el9_gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00

source /scratch/kaustubh/miniconda3/bin/activate bev_env
cd /scratch/kaustubh/bevformer_project/BEVFormer

export PYTHONPATH=$PYTHONPATH:/scratch/kaustubh/bevformer_project/mmdetection3d:.

python -u tools/test.py projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    --out /scratch/kaustubh/bevformer_project/BEVFormer/custom_car_predictions.pkl \
    --cfg-options \
    data.test.ann_file=custom_nuscenes_infos.pkl \
    data.test.test_mode=True \
    data.test.filter_empty_gt=False \
    data.workers_per_gpu=0
