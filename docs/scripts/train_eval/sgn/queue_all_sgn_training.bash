#!/bin/bash

echo "🚀 Queueing All SGN Training Jobs"
echo "=" * 50

# Create logs directory
mkdir -p logs

# Define all required jobs
# AR (Action Recognition) - All datasets and settings
echo "📝 Creating SGN AR jobs..."

# SGN AR NTU CS
echo "Creating SGN AR NTU CS job..."
python train_sgn.py --dataset ntu --setting cs --task ar --slurm
if [ -f "bash/train_eval/sgn/train_sgn_ar_ntu_cs.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_ar_ntu_cs.sbatch
    echo "✅ Submitted SGN AR NTU CS"
else
    echo "❌ Failed to create SGN AR NTU CS script"
fi

# SGN AR NTU CV
echo "Creating SGN AR NTU CV job..."
python train_sgn.py --dataset ntu --setting cv --task ar --slurm
if [ -f "bash/train_eval/sgn/train_sgn_ar_ntu_cv.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_ar_ntu_cv.sbatch
    echo "✅ Submitted SGN AR NTU CV"
else
    echo "❌ Failed to create SGN AR NTU CV script"
fi

# SGN AR NTU120 CS
echo "Creating SGN AR NTU120 CS job..."
python train_sgn.py --dataset ntu120 --setting cs --task ar --slurm
if [ -f "bash/train_eval/sgn/train_sgn_ar_ntu120_cs.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_ar_ntu120_cs.sbatch
    echo "✅ Submitted SGN AR NTU120 CS"
else
    echo "❌ Failed to create SGN AR NTU120 CS script"
fi

# SGN AR NTU120 CV
echo "Creating SGN AR NTU120 CV job..."
python train_sgn.py --dataset ntu120 --setting cv --task ar --slurm
if [ -f "bash/train_eval/sgn/train_sgn_ar_ntu120_cv.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_ar_ntu120_cv.sbatch
    echo "✅ Submitted SGN AR NTU120 CV"
else
    echo "❌ Failed to create SGN AR NTU120 CV script"
fi

echo ""
echo "📝 Creating SGN RI jobs (CV only)..."

# SGN RI NTU CV
echo "Creating SGN RI NTU CV job..."
python train_sgn.py --dataset ntu --setting cv --task ri --slurm
if [ -f "bash/train_eval/sgn/train_sgn_ri_ntu_cv.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_ri_ntu_cv.sbatch
    echo "✅ Submitted SGN RI NTU CV"
else
    echo "❌ Failed to create SGN RI NTU CV script"
fi

# SGN RI NTU120 CV
echo "Creating SGN RI NTU120 CV job..."
python train_sgn.py --dataset ntu120 --setting cv --task ri --slurm
if [ -f "bash/train_eval/sgn/train_sgn_ri_ntu120_cv.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_ri_ntu120_cv.sbatch
    echo "✅ Submitted SGN RI NTU120 CV"
else
    echo "❌ Failed to create SGN RI NTU120 CV script"
fi

echo ""
echo "📝 Creating SGN GC jobs..."

# SGN GC NTU CS
echo "Creating SGN GC NTU CS job..."
python train_sgn.py --dataset ntu --setting cs --task gc --slurm
if [ -f "bash/train_eval/sgn/train_sgn_gc_ntu_cs.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_gc_ntu_cs.sbatch
    echo "✅ Submitted SGN GC NTU CS"
else
    echo "❌ Failed to create SGN GC NTU CS script"
fi

# SGN GC NTU CV
echo "Creating SGN GC NTU CV job..."
python train_sgn.py --dataset ntu --setting cv --task gc --slurm
if [ -f "bash/train_eval/sgn/train_sgn_gc_ntu_cv.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_gc_ntu_cv.sbatch
    echo "✅ Submitted SGN GC NTU CV"
else
    echo "❌ Failed to create SGN GC NTU CV script"
fi

# SGN GC NTU120 CS
echo "Creating SGN GC NTU120 CS job..."
python train_sgn.py --dataset ntu120 --setting cs --task gc --slurm
if [ -f "bash/train_eval/sgn/train_sgn_gc_ntu120_cs.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_gc_ntu120_cs.sbatch
    echo "✅ Submitted SGN GC NTU120 CS"
else
    echo "❌ Failed to create SGN GC NTU120 CS script"
fi

# SGN GC NTU120 CV
echo "Creating SGN GC NTU120 CV job..."
python train_sgn.py --dataset ntu120 --setting cv --task gc --slurm
if [ -f "bash/train_eval/sgn/train_sgn_gc_ntu120_cv.sbatch" ]; then
    sbatch bash/train_eval/sgn/train_sgn_gc_ntu120_cv.sbatch
    echo "✅ Submitted SGN GC NTU120 CV"
else
    echo "❌ Failed to create SGN GC NTU120 CV script"
fi

echo ""
echo "🎉 All SGN training jobs queued!"
echo ""
echo "💡 To check job status: squeue -u $USER"
echo "💡 To check logs: ls -la logs/sgn_*"

