#!/bin/bash
# ============================================================================
# MIRAGE-Enhanced TMR Training Pipeline
# ============================================================================
# Submits SLURM jobs for:
#   1. Full combo (all 5 MIRAGE losses)
#   2. Individual ablations (each loss alone)
# Each experiment chains: Stage 3 train -> retarget -> downstream SGN eval
# ============================================================================

set -euo pipefail

PROJECT_DIR="/users/tcarr23/Transformer-Retargeting"
CHECKPOINT="output/disentangled_tmr_stable/checkpoint_stage3_best.pth"
DATA_PATH="data/ntu_cv_paired_10000_2000.pt"
LOG_DIR="logs/mirage_enhanced"
OUTPUT_BASE="output/mirage_enhanced"

mkdir -p "${PROJECT_DIR}/${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/${OUTPUT_BASE}"

# Common SLURM settings
PARTITION="GPU"
TIME="14:00:00"
MEM="32G"
GPUS="1"
CPUS="8"

# Common training args
COMMON_ARGS="--data_path ${DATA_PATH} --dataset ntu --stage3_epochs 30 \
  --batch_size 32 --lr 5e-4 --use_gradient_clip \
  --no_wandb --save_freq 10 --log_freq 20 --early_stop_patience 20 \
  --resume ${CHECKPOINT}"

# ============================================================================
# Helper: submit a chained pipeline (train -> retarget -> downstream eval)
# ============================================================================
submit_pipeline() {
    local NAME="$1"
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    local MIRAGE_FLAGS="$2"
    local RETARGETED_PATH="${OUTPUT_DIR}/retargeted_ntu.pkl"

    mkdir -p "${PROJECT_DIR}/${OUTPUT_DIR}"

    # --- Job 1: Stage 3 Training ---
    TRAIN_SCRIPT=$(mktemp "${PROJECT_DIR}/sbatch_queue/train_${NAME}_XXXXXX.sh")
    cat > "${TRAIN_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=tmr_m_${NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEM}
#SBATCH --exclude=str-gpu4,str-gpu5
#SBATCH --output=${LOG_DIR}/${NAME}_train_%j.out
#SBATCH --error=${LOG_DIR}/${NAME}_train_%j.err

module load pytorch/2.3.0-cuda12.1
cd ${PROJECT_DIR}
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH

echo "=== Training: ${NAME} ==="
echo "MIRAGE flags: ${MIRAGE_FLAGS}"

python scripts/train_stage3_mirage.py \\
    ${COMMON_ARGS} \\
    --output_dir ${OUTPUT_DIR} \\
    --use_mirage_losses \\
    ${MIRAGE_FLAGS}

echo "=== Training complete ==="
EOF

    TRAIN_JOB=$(sbatch "${TRAIN_SCRIPT}" | awk '{print $4}')
    echo "  [${NAME}] Train job: ${TRAIN_JOB}"

    # --- Job 2: Generate retargeted dataset ---
    RETARGET_SCRIPT=$(mktemp "${PROJECT_DIR}/sbatch_queue/retarget_${NAME}_XXXXXX.sh")
    cat > "${RETARGET_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=tmr_r_${NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=${LOG_DIR}/${NAME}_retarget_%j.out
#SBATCH --error=${LOG_DIR}/${NAME}_retarget_%j.err

module load pytorch/2.3.0-cuda12.1
cd ${PROJECT_DIR}
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH

echo "=== Retargeting: ${NAME} ==="
python scripts/generate_retargeted_dataset.py \\
    --checkpoint ${OUTPUT_DIR}/checkpoint_stage3_best.pth \\
    --dataset ntu \\
    --output_path ${RETARGETED_PATH}

echo "=== Retargeting complete ==="
EOF

    RETARGET_JOB=$(sbatch --dependency=afterok:${TRAIN_JOB} "${RETARGET_SCRIPT}" | awk '{print $4}')
    echo "  [${NAME}] Retarget job: ${RETARGET_JOB} (depends on ${TRAIN_JOB})"

    # --- Job 3: Downstream SGN eval (AR only) ---
    EVAL_AR_SCRIPT=$(mktemp "${PROJECT_DIR}/sbatch_queue/eval_ar_${NAME}_XXXXXX.sh")
    cat > "${EVAL_AR_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=tmr_ea_${NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=${LOG_DIR}/${NAME}_eval_ar_%j.out
#SBATCH --error=${LOG_DIR}/${NAME}_eval_ar_%j.err

module load pytorch/2.3.0-cuda12.1
cd ${PROJECT_DIR}
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH

echo "=== Downstream SGN AR: ${NAME} ==="
python scripts/train_downstream_models.py \\
    --dataset ntu --setting cv \\
    --data_path ${RETARGETED_PATH} \\
    --models sgn_ar \\
    --epochs 60 --batch_size 128 \\
    --output_root ${OUTPUT_DIR}

echo "=== SGN AR eval complete ==="
EOF

    EVAL_AR_JOB=$(sbatch --dependency=afterok:${RETARGET_JOB} "${EVAL_AR_SCRIPT}" | awk '{print $4}')
    echo "  [${NAME}] SGN AR eval job: ${EVAL_AR_JOB} (depends on ${RETARGET_JOB})"

    # --- Job 4: Downstream SGN eval (RI only) ---
    EVAL_RI_SCRIPT=$(mktemp "${PROJECT_DIR}/sbatch_queue/eval_ri_${NAME}_XXXXXX.sh")
    cat > "${EVAL_RI_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=tmr_er_${NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=${LOG_DIR}/${NAME}_eval_ri_%j.out
#SBATCH --error=${LOG_DIR}/${NAME}_eval_ri_%j.err

module load pytorch/2.3.0-cuda12.1
cd ${PROJECT_DIR}
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH

echo "=== Downstream SGN RI: ${NAME} ==="
python scripts/train_downstream_models.py \\
    --dataset ntu --setting cv \\
    --data_path ${RETARGETED_PATH} \\
    --models sgn_ri \\
    --epochs 60 --batch_size 128 \\
    --output_root ${OUTPUT_DIR}

echo "=== SGN RI eval complete ==="
EOF

    EVAL_RI_JOB=$(sbatch --dependency=afterok:${RETARGET_JOB} "${EVAL_RI_SCRIPT}" | awk '{print $4}')
    echo "  [${NAME}] SGN RI eval job: ${EVAL_RI_JOB} (depends on ${RETARGET_JOB})"

    echo "  [${NAME}] Pipeline submitted: ${TRAIN_JOB} -> ${RETARGET_JOB} -> {${EVAL_AR_JOB}, ${EVAL_RI_JOB}}"
}

# ============================================================================
# Submit experiments
# ============================================================================

echo ""
echo "============================================"
echo " MIRAGE-Enhanced TMR Experiments"
echo "============================================"
echo ""

# 1. Full combo: all 5 MIRAGE losses
echo "--- Full Combo (all losses) ---"
submit_pipeline "full_combo" \
    "--lambda_dist_disc 1.0 --lambda_output_act 1.0 --lambda_output_id 1.0 --lambda_output_contrastive 1.0 --lambda_ee_enhanced 1.0"

echo ""

# 2. Ablation: Distribution discriminator only
echo "--- Ablation: dist_disc only ---"
submit_pipeline "abl_dist_disc" \
    "--lambda_dist_disc 1.0 --lambda_output_act 0.0 --lambda_output_id 0.0 --lambda_output_contrastive 0.0 --lambda_ee_enhanced 0.0"

echo ""

# 3. Ablation: Output action classifier only
echo "--- Ablation: output_act only ---"
submit_pipeline "abl_output_act" \
    "--lambda_dist_disc 0.0 --lambda_output_act 1.0 --lambda_output_id 0.0 --lambda_output_contrastive 0.0 --lambda_ee_enhanced 0.0"

echo ""

# 4. Ablation: Output identity adversary only
echo "--- Ablation: output_id only ---"
submit_pipeline "abl_output_id" \
    "--lambda_dist_disc 0.0 --lambda_output_act 0.0 --lambda_output_id 1.0 --lambda_output_contrastive 0.0 --lambda_ee_enhanced 0.0"

echo ""

# 5. Ablation: Output contrastive only
echo "--- Ablation: output_contrastive only ---"
submit_pipeline "abl_contrastive" \
    "--lambda_dist_disc 0.0 --lambda_output_act 0.0 --lambda_output_id 0.0 --lambda_output_contrastive 1.0 --lambda_ee_enhanced 0.0"

echo ""

# 6. Ablation: Enhanced EE only
echo "--- Ablation: ee_enhanced only ---"
submit_pipeline "abl_ee_enhanced" \
    "--lambda_dist_disc 0.0 --lambda_output_act 0.0 --lambda_output_id 0.0 --lambda_output_contrastive 0.0 --lambda_ee_enhanced 1.0"

echo ""
echo "============================================"
echo " All experiments submitted!"
echo "============================================"
