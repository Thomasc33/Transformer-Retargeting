#!/bin/bash
#SBATCH --job-name=test_training_1gpu
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/test_training_1gpu_%j.out
#SBATCH --error=logs/test_training_1gpu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

module load pytorch/2.3.0-cuda12.1

python -u main.py \
    --dataset ntu \
    --setting cv \
    --epochs 15 \
    --batch-size 32 \
    --lr 9.43062936149491e-05 \
    --train-samples 2000 \
    --test-samples 250 \
    --config configs/main_config.yaml \
    --gradient-accumulation-steps 1 \
    --mixed-precision \
    --validate-every 3 \
    --progress-every 50 \
    --use-checkpoint \
    --decoder-dropout 0.11551063114920847 \
    --save-every 5 \
    --wandb-project "Motion Retargeting Test 1GPU" \
    --loss-weights mse:5.323284271000699,ee:4.7250245075017725,smoothing:2.8747697025246937,inception:3.0656068713914353,fid_vel:2.2608337791442894,bone:6.185377286837532,foot:0.544125368059208,joint_limit:1.768344443841404

