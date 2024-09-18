#!/bin/bash
# Post-hoc OA sweep: generate retargeted data from OA checkpoint at each beta (no retraining)
# Then cross-evaluate with frozen SGN
set -e

BETAS="0.03 0.05 0.08 0.10 0.15 0.20 0.25 0.30 0.50 0.70"
CKPT="output/mirage_enhanced/abl_output_act/checkpoint_stage3_best.pth"
DOWNSTREAM_ROOT="output/downstream_ntu60_raw"

mkdir -p sbatch_queue/beta_posthoc_oa logs/beta_posthoc_oa

for BETA in $BETAS; do
    BETA_DIR=$(echo $BETA | tr '.' '_')
    OUTDIR="output/beta_posthoc_oa/beta_${BETA_DIR}"
    SCRIPT="sbatch_queue/beta_posthoc_oa/gen_eval_${BETA_DIR}.sh"

    cat > "$SCRIPT" << SLURM
#!/bin/bash
#SBATCH --job-name=phoa_${BETA_DIR}
#SBATCH --partition=GPU
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/beta_posthoc_oa/${BETA_DIR}_%j.out
#SBATCH --error=logs/beta_posthoc_oa/${BETA_DIR}_%j.err

module load pytorch/2.3.0-cuda12.1
cd /users/tcarr23/Transformer-Retargeting
export PYTHONPATH=/users/tcarr23/Transformer-Retargeting:\$PYTHONPATH

echo "=== Generate retargeted dataset: OA checkpoint, beta=${BETA} (no retrain) ==="
python scripts/generate_retargeted_dataset.py \\
    --checkpoint ${CKPT} \\
    --dataset ntu \\
    --output_path ${OUTDIR}/retargeted_ntu.pkl \\
    --beta ${BETA}

echo "=== Cross-evaluate with frozen SGN ==="
python scripts/cross_evaluate_downstream.py \\
    --checkpoint_root ${DOWNSTREAM_ROOT} \\
    --raw_data_path ${OUTDIR}/retargeted_ntu.pkl \\
    --dataset ntu --setting cv \\
    --output_dir ${OUTDIR} \\
    --models sgn_ar sgn_ri

echo "=== Done: OA posthoc beta=${BETA} ==="
SLURM

    sbatch "$SCRIPT"
done

echo "All OA post-hoc jobs submitted."
