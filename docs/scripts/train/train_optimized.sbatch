#!/bin/bash
#SBATCH --job-name=motion_retarget_optimized
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/training_optimized_%j.out
#SBATCH --error=logs/training_optimized_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

module load pytorch/2.3.0-cuda12.1

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    main.py \
    --hpc \
    --dataset ntu \
    --setting cv \
    --epochs 5 \
    --batch-size 8 \
    --lr 9.43062936149491e-05 \
    --train-samples 999999999 \
    --test-samples 10000 \
    --config configs/main_config.yaml \
    --gradient-accumulation-steps 4 \
    --mixed-precision \
    --validate-every 5 \
    --progress-every 100 \
    --use-checkpoint \
    --nccl-timeout 7200 \
    --decoder-dropout 0.11551063114920847 \
    --save-every 1 \
    --loss-weights mse:5.323284271000699,ee:4.7250245075017725,smoothing:2.8747697025246937,inception:3.0656068713914353,fid_vel:2.2608337791442894,bone:6.185377286837532,foot:0.544125368059208,joint_limit:1.768344443841404

