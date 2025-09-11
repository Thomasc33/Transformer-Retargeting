#!/bin/bash
#SBATCH --job-name=pretrained_unfrozen
#SBATCH --output=slurm_out/pretraining/pretrained_unfrozen_%j.out
#SBATCH --error=slurm_out/pretraining/pretrained_unfrozen_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --time=240:00:00

# Create output directories if they don't exist
mkdir -p slurm_out/pretraining
mkdir -p experiments/pretraining/results/pretrained_unfrozen

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Print CUDA availability
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Run training with pretrained encoder but without freezing
torchrun --nproc_per_node=1 --master_port=29502 main.py \
    --dataset=ntu \
    --setting=cv \
    --batch-size=128 \
    --lr=1e-5 \
    --epochs=100 \
    --train-samples=10000 \
    --test-samples=2000 \
    --teacher-forcing-ratio=1.0 \
    --teacher-forcing-decay=0.0 \
    --loss-mse=7.0 \
    --loss-ee=5.0 \
    --loss-smoothing=0.075 \
    --loss-inception=0.05 \
    --loss-fid-vel=1.0 \
    --loss-bone=10.0 \
    --loss-foot=3.0 \
    --loss-joint-limit=1.0 \
    --decoder-dropout=0.1 \
    --output-model-path=experiments/pretraining/results/pretrained_unfrozen/model.pth \
    --use-pretrained \
    --no-freeze-encoder \
    --run-eval \
    --hpc

# Run a more detailed evaluation after training
python eval_model.py \
    --dataset=ntu \
    --setting=cv \
    --model_type=transformer \
    --transformer_model_path=experiments/pretraining/results/pretrained_unfrozen/model.pth \
    --eval_model=sgn \
    --test_samples=2000 \
    --output_dir=experiments/pretraining/results/pretrained_unfrozen \
    --ar_model_weights=eval/sgn/pretrained/ntu/cview_ar.pth \
    --ri_model_weights=eval/sgn/pretrained/ntu/cview_ri.pth \
    --gc_model_weights=output/ntu_gc_cview/NTU_gc_cview/model_best.pth.tar

echo "Experiment completed: Pretrained encoder without freezing"
