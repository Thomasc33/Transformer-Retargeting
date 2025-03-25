#!/bin/bash
#
#SBATCH --job-name="mixformer"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#
#   ===== Main =====


module load pytorch/2.3.0-cuda12.1

echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

python eval.py \
    --dataset ntu \
    --eval_model mixformer \
    --transformer_model_path model.pth \
    --ar_model_weights eval/mixformer/pretrained/ntu/ar.pth \
    --ri_model_weights eval/mixformer/pretrained/ntu/ri.pth \
    --batch_size 32 \
    --paired_batch_size 8 \
    --test_samples 2000