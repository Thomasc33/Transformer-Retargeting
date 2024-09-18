#!/bin/bash
# Full beta sweep: retrain → generate → cross-evaluate (matches old pipeline)
# Uses BASELINE checkpoint (disentangled_tmr_stable) + output_act variant
set -e

BETAS="0.03 0.05 0.08 0.10 0.15 0.20 0.25 0.30 0.50 0.70"
DATA_PATH="data/ntu_cv_paired_10k.pt"
DOWNSTREAM_ROOT="output/downstream_ntu60_raw"

mkdir -p sbatch_queue/beta_full logs/beta_full

for BASE in baseline output_act; do
    if [ "$BASE" = "baseline" ]; then
        CKPT="output/disentangled_tmr_stable/checkpoint_stage3_best.pth"
        MIRAGE_FLAGS="--lambda_dist_disc 0.0 --lambda_output_act 0.0 --lambda_output_id 0.0 --lambda_output_contrastive 0.0 --lambda_ee_enhanced 0.0"
    else
        CKPT="output/mirage_enhanced/abl_output_act/checkpoint_stage3_best.pth"
        MIRAGE_FLAGS="--lambda_dist_disc 0.0 --lambda_output_act 1.0 --lambda_output_id 0.0 --lambda_output_contrastive 0.0 --lambda_ee_enhanced 0.0"
    fi

    for BETA in $BETAS; do
        BETA_DIR=$(echo $BETA | tr '.' '_')
        OUTDIR="output/beta_full/${BASE}_beta_${BETA_DIR}"
        SCRIPT="sbatch_queue/beta_full/${BASE}_beta_${BETA_DIR}.sh"

        cat > "$SCRIPT" << SLURM
#!/bin/bash
#SBATCH --job-name=bf_${BASE:0:1}${BETA_DIR}
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/beta_full/${BASE}_beta_${BETA_DIR}_%j.out
#SBATCH --error=logs/beta_full/${BASE}_beta_${BETA_DIR}_%j.err

module load pytorch/2.3.0-cuda12.1
cd /users/tcarr23/Transformer-Retargeting
export PYTHONPATH=/users/tcarr23/Transformer-Retargeting:\$PYTHONPATH

OUTDIR=${OUTDIR}

echo "=== Step 1: Retrain Stage 3 with beta=${BETA} (base=${BASE}) ==="
python scripts/train_stage3_mirage.py \\
    --data_path ${DATA_PATH} --dataset ntu \\
    --beta ${BETA} \\
    --stage3_epochs 30 --batch_size 32 --lr 5e-4 \\
    --use_gradient_clip --no_wandb --save_freq 5 --log_freq 20 \\
    --early_stop_patience 20 \\
    --resume ${CKPT} \\
    --output_dir \$OUTDIR \\
    --use_mirage_losses \\
    ${MIRAGE_FLAGS}

echo "=== Step 2: Generate retargeted dataset ==="
python scripts/generate_retargeted_dataset.py \\
    --checkpoint \$OUTDIR/checkpoint_stage3_best.pth \\
    --dataset ntu \\
    --output_path \$OUTDIR/retargeted_ntu.pkl \\
    --beta ${BETA}

echo "=== Step 3: Cross-evaluate with frozen models ==="
python scripts/cross_evaluate_downstream.py \\
    --checkpoint_root ${DOWNSTREAM_ROOT} \\
    --raw_data_path \$OUTDIR/retargeted_ntu.pkl \\
    --dataset ntu --setting cv \\
    --output_dir \$OUTDIR \\
    --models sgn_ar sgn_ri

echo "=== Done: ${BASE} beta=${BETA} ==="
SLURM

        echo "Submitting ${BASE} beta=${BETA}..."
        sbatch "$SCRIPT"
    done
done

echo "All jobs submitted."
