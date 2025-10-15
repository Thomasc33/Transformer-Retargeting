#!/bin/bash

# MLM Feature-Based Evaluation - Quick Start Script
# This script provides easy commands to run the comprehensive MLM evaluation

echo "============================================================"
echo "MLM FEATURE-BASED EVALUATION SUITE"
echo "============================================================"
echo ""
echo "This evaluation uses encoder features for classification instead"
echo "of reconstructed coordinates, addressing the fundamental issue"
echo "with previous MLM evaluation (AR ~2%, RI ~4%)."
echo ""
echo "Expected new results: AR 60-85%, RI 40-70%"
echo ""

# Check if we're in the right directory
if [ ! -f "evaluation_suite/queue_mlm_classification_jobs.bash" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "Available commands:"
echo ""
echo "0. DEBUG STEP-BY-STEP (IF HAVING ISSUES):"
echo "   python evaluation_suite/debug_step_by_step.py"
echo ""
echo "1. TEST SINGLE EVALUATION (RECOMMENDED FIRST):"
echo "   python evaluation_suite/run_single_mlm_test.py"
echo ""
echo "2. QUEUE ALL 9 JOBS:"
echo "   ./evaluation_suite/queue_mlm_classification_jobs.bash --dataset ntu --setting cv"
echo ""
echo "3. QUEUE WITH CUSTOM SETTINGS:"
echo "   ./evaluation_suite/queue_mlm_classification_jobs.bash --dataset ntu120 --setting cs"
echo ""
echo "4. RUN SINGLE CONFIGURATION (MANUAL):"
echo "   python evaluation_suite/comprehensive_mlm_evaluation.py \\"
echo "       --model-dir eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.3 \\"
echo "       --dataset ntu --setting cv --temporal-ratio 0.3 --spatial-ratio 0.3 \\"
echo "       --output-dir results/comprehensive_mlm_evaluation"
echo ""
echo "5. GENERATE REPORTS (AFTER JOBS COMPLETE):"
echo "   python evaluation_suite/generate_comprehensive_mlm_report.py \\"
echo "       --results-dir results/comprehensive_mlm_evaluation \\"
echo "       --dataset ntu --setting cv \\"
echo "       --output-dir results/comprehensive_mlm_evaluation/reports"
echo ""
echo "6. MONITOR JOBS:"
echo "   squeue -u \$USER"
echo ""
echo "7. CHECK RESULTS:"
echo "   ls results/comprehensive_mlm_evaluation/"
echo "   ls results/comprehensive_mlm_evaluation/reports/"
echo ""

# Parse command line arguments
if [ "$1" = "--debug" ]; then
    echo "============================================================"
    echo "RUNNING MLM EVALUATION DEBUG"
    echo "============================================================"
    echo ""

    echo "Running step-by-step debugging..."
    python evaluation_suite/debug_step_by_step.py

elif [ "$1" = "--test" ]; then
    echo "============================================================"
    echo "RUNNING MLM EVALUATION TEST"
    echo "============================================================"
    echo ""

    echo "Running single evaluation test..."
    python evaluation_suite/run_single_mlm_test.py

elif [ "$1" = "--run" ]; then
    echo "============================================================"
    echo "RUNNING MLM EVALUATION FOR NTU CROSS-VIEW"
    echo "============================================================"
    echo ""

    # Queue all jobs
    echo "Queuing all 9 MLM evaluation jobs..."
    ./evaluation_suite/queue_mlm_classification_jobs.bash --dataset ntu --setting cv

    echo ""
    echo "Jobs queued! Monitor with: squeue -u \$USER"
    echo "Results will be available in: results/comprehensive_mlm_evaluation/"
    echo ""
    echo "The summary job will automatically generate reports when all jobs complete."
    echo "Estimated total time: ~12 hours"

elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage:"
    echo "  $0                 # Show available commands"
    echo "  $0 --debug         # Run step-by-step debugging"
    echo "  $0 --test          # Run single evaluation test"
    echo "  $0 --run           # Queue all NTU cross-view jobs"
    echo "  $0 --help          # Show this help"

else
    echo "To debug issues, use:"
    echo "  $0 --debug"
    echo ""
    echo "To test a single evaluation, use:"
    echo "  $0 --test"
    echo ""
    echo "To run the full evaluation, use:"
    echo "  $0 --run"
    echo ""
    echo "Or copy and paste one of the commands above."
fi

echo ""
echo "============================================================"
echo "For more details, see: evaluation_suite/MLM_EVALUATION_README.md"
echo "============================================================"
