#!/bin/bash
# NTU120 pipeline (matches ETRI training): train OA → generate (beta=0.2) → downstream SGN+MixFormer
# Two chained SLURM jobs: downstream starts as soon as retargeted data is ready.
set -e

mkdir -p sbatch_queue/ntu120 logs/ntu120

# ─── Job 1: Train NTU120 OA model + generate retargeted dataset ───────────────
cat > sbatch_queue/ntu120/train_and_gen.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=ntu120_train
#SBATCH --partition=GPU
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/ntu120/train_%j.out
#SBATCH --error=logs/ntu120/train_%j.err

module load pytorch/2.3.0-cuda12.1
cd /users/tcarr23/Transformer-Retargeting
export PYTHONPATH=/users/tcarr23/Transformer-Retargeting:$PYTHONPATH

OUTDIR="output/ntu120_oa_b02"

echo "=== Step 1: Train DisentangledTMR (OA) on NTU120 ==="
python scripts/train_disentangled_tmr.py \
    --data_path data/ntu120_cv_paired_10000_2000.pt \
    --dataset ntu120 \
    --output_dir $OUTDIR/model \
    --stage1_epochs 20 \
    --stage2_epochs 15 \
    --stage3_epochs 20 \
    --batch_size 32 \
    --no_amp \
    --no_lstm \
    --seed 42 \
    --auto_resume \
    --weight_orthogonality 0.1 \
    --weight_adversarial 0.5 \
    --weight_end_effector 2.0 \
    --weight_bone_length 1.0 \
    --weight_motion_dynamics 0.2 \
    --weight_temporal_smoothness 0.1 \
    --lambda_dist_disc 0.0 \
    --lambda_output_act 1.0 \
    --lambda_output_id 0.0 \
    --lambda_output_contrastive 0.0 \
    --lambda_ee_enhanced 0.0 \
    --no_wandb

echo "=== Step 2: Generate retargeted NTU120 (post-hoc beta=0.2) ==="
python scripts/generate_retargeted_dataset.py \
    --checkpoint $OUTDIR/model/checkpoint_stage3_best.pth \
    --dataset ntu120 \
    --output_path $OUTDIR/retargeted_ntu120_b02.pkl \
    --seed 42 \
    --beta 0.2

echo "=== Train+generate done: $(date) ==="
SLURM

# ─── Job 2: Train downstream SGN + MixFormer on retargeted NTU120 ─────────────
cat > sbatch_queue/ntu120/downstream.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=ntu120_down
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/ntu120/downstream_%j.out
#SBATCH --error=logs/ntu120/downstream_%j.err

module load pytorch/2.3.0-cuda12.1
cd /users/tcarr23/Transformer-Retargeting
export PYTHONPATH=/users/tcarr23/Transformer-Retargeting:$PYTHONPATH

OUTDIR="output/ntu120_oa_b02"

echo "=== Step 3: Train downstream models on retargeted NTU120 (beta=0.2) ==="
python scripts/train_downstream_models.py \
    --dataset ntu120 \
    --data_path $OUTDIR/retargeted_ntu120_b02.pkl \
    --setting cv \
    --output_root $OUTDIR/downstream \
    --models sgn_ar sgn_ri mix_ar mix_ri \
    --epochs 60 \
    --batch_size 128 \
    --num_workers 4

echo ""
echo "=== Downstream done: $(date) ==="
echo "Metrics:"
cat $OUTDIR/downstream/metrics.json 2>/dev/null || echo "(no metrics file)"
SLURM

# Submit with dependency
JOB1=$(sbatch --parsable sbatch_queue/ntu120/train_and_gen.sh)
echo "Job 1 (train+generate): $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 sbatch_queue/ntu120/downstream.sh)
echo "Job 2 (downstream):     $JOB2 (starts after $JOB1)"

echo ""
echo "Monitor: squeue -u $USER -j $JOB1,$JOB2"
echo "Logs:    logs/ntu120/train_${JOB1}.out"
echo "         logs/ntu120/downstream_${JOB2}.out"
