#!/bin/bash
# MixFormer on the main partial-retargeting result (seed2, β=0.2).
#
# Covers the missing piece of the ground-truth table:
#   (a) frozen MixFormer cross-eval on retargeted seed2 data
#   (b) retrained MixFormer AR/RI on retargeted seed2 data
#
# The seed2 checkpoint is the reference result (76.2% SGN AR / 14.9% SGN RI).
set -e

CKPT="output/beta_improve/mirage_full_seed2/checkpoint_stage3_best.pth"
OUTDIR="output/round2/seed2_b02_mixformer"
RETARGETED="${OUTDIR}/retargeted_ntu_seed2_b02.pkl"
DOWNSTREAM_ROOT="output/downstream_ntu60_raw"

mkdir -p sbatch_queue/seed2_mix logs/seed2_mix "${OUTDIR}"

# Job 1: Generate retargeted data + frozen MixFormer cross-eval
cat > sbatch_queue/seed2_mix/gen_cross.sh << SLURM
#!/bin/bash
#SBATCH --job-name=s2mix_gen
#SBATCH --partition=GPU
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/seed2_mix/gen_cross_%j.out
#SBATCH --error=logs/seed2_mix/gen_cross_%j.err

module load pytorch/2.3.0-cuda12.1
cd /users/tcarr23/Transformer-Retargeting
export PYTHONPATH=/users/tcarr23/Transformer-Retargeting:\$PYTHONPATH

echo "=== [\$(date)] Generate retargeted dataset: seed2 ckpt, beta=0.2 ==="
python scripts/generate_retargeted_dataset.py \\
    --checkpoint ${CKPT} \\
    --dataset ntu \\
    --output_path ${RETARGETED} \\
    --beta 0.2 \\
    --batch_size 64 \\
    --seed 2

echo "=== [\$(date)] Cross-evaluate frozen SGN + MixFormer ==="
python scripts/cross_evaluate_downstream.py \\
    --checkpoint_root ${DOWNSTREAM_ROOT} \\
    --raw_data_path ${RETARGETED} \\
    --dataset ntu --setting cv \\
    --output_dir ${OUTDIR} \\
    --models sgn_ar sgn_ri mix_ar mix_ri

echo "=== [\$(date)] Frozen cross-eval done ==="
cat ${OUTDIR}/cross_eval_metrics.json
SLURM

# Job 2: Retrain MixFormer (and SGN for sanity) on retargeted data
cat > sbatch_queue/seed2_mix/retrain_mix.sh << SLURM
#!/bin/bash
#SBATCH --job-name=s2mix_retr
#SBATCH --partition=GPU
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/seed2_mix/retrain_mix_%j.out
#SBATCH --error=logs/seed2_mix/retrain_mix_%j.err

module load pytorch/2.3.0-cuda12.1
cd /users/tcarr23/Transformer-Retargeting
export PYTHONPATH=/users/tcarr23/Transformer-Retargeting:\$PYTHONPATH

echo "=== [\$(date)] Retrain MixFormer (and SGN) on seed2 retargeted data ==="
python scripts/train_downstream_models.py \\
    --dataset ntu \\
    --data_path ${RETARGETED} \\
    --setting cv \\
    --output_root ${OUTDIR}/retrained \\
    --models sgn_ar mix_ar \\
    --epochs 60 \\
    --batch_size 128 \\
    --num_workers 4

echo "=== [\$(date)] Retrain done ==="
cat ${OUTDIR}/retrained/metrics.json
SLURM

JOB1=$(sbatch --parsable sbatch_queue/seed2_mix/gen_cross.sh)
echo "Job 1 (gen + frozen cross-eval): $JOB1"
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 sbatch_queue/seed2_mix/retrain_mix.sh)
echo "Job 2 (retrain MixFormer):       $JOB2 (starts after $JOB1)"

echo ""
echo "Monitor: squeue -u $USER -j $JOB1,$JOB2"
