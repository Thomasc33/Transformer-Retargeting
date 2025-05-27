#!/bin/bash
#
#SBATCH --job-name="mixformer"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#
#   ===== Main =====


module load pytorch/2.3.0-cuda12.1

echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

cd ..
cd ..

python eval_model.py \
    --dataset ntu \
    --eval_model sgn \
    --model_type pmr \
    --setting cv \
    --batch_size 32 \
    --paired_batch_size 8 \
    --test_samples 2000 \
    --T 75 \
    --output_dir results/pmr_sgn \
    --ar_model_weights eval/sgn/pretrained/ntu/cview_ar.pth \
    --ri_model_weights eval/sgn/pretrained/ntu/cview_ri.pth \
    --gc_model_weights output/ntu_gc_cview/NTU_gc_cview/model_best.pth.tar