#!/bin/bash
# Beta sweep: post-hoc blending on abl_output_act with source+target RI
set -e

BETAS="0.03 0.05 0.08 0.10 0.15 0.20 0.25 0.30 0.50 0.70 1.00"
CKPT="output/mirage_enhanced/abl_output_act/checkpoint_stage3_best.pth"
DATA_PATH="data/ntu_cv_paired_10k.pt"
SGN_AR="output/downstream_ntu60_raw/ntu_sgn_ar_paired/model_best.pth.tar"
SGN_RI="output/downstream_ntu60_raw/ntu_sgn_ri_paired/model_best.pth.tar"
MIX_AR="output/downstream_ntu60_raw/ntu_mixformer_ar_paired/model_best.pth.tar"
MIX_RI="output/downstream_ntu60_raw/ntu_mixformer_ri_paired/model_best.pth.tar"

mkdir -p sbatch_queue/beta_sweep logs/beta_sweep

for BETA in $BETAS; do
    BETA_DIR=$(echo $BETA | tr '.' '_')
    SCRIPT="sbatch_queue/beta_sweep/v2_beta_${BETA_DIR}.sh"

    cat > "$SCRIPT" << SLURM
#!/bin/bash
#SBATCH --job-name=bv2_${BETA_DIR}
#SBATCH --partition=GPU
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/beta_sweep/v2_${BETA_DIR}_%j.out
#SBATCH --error=logs/beta_sweep/v2_${BETA_DIR}_%j.err

module load pytorch/2.3.0-cuda12.1
cd /users/tcarr23/Transformer-Retargeting
export PYTHONPATH=/users/tcarr23/Transformer-Retargeting:\$PYTHONPATH

python scripts/eval_tmr_ablation.py \\
    --checkpoint ${CKPT} \\
    --data_path ${DATA_PATH} --dataset ntu \\
    --beta ${BETA} \\
    --sgn_ar_ckpt ${SGN_AR} --sgn_ri_ckpt ${SGN_RI} \\
    --mix_ar_ckpt ${MIX_AR} --mix_ri_ckpt ${MIX_RI} \\
    --batch_size 32 \\
    --include_baselines \\
    --output logs/beta_sweep/v2_${BETA_DIR}.json
SLURM

    sbatch "$SCRIPT"
done
echo "All jobs submitted."
