# Evaluation Results Reference

This document provides a comprehensive reference to all evaluation results, log files, and output locations for the Transformer-Retargeting project.

**Last Updated**: 2025-08-27 11:01:00

## 🚀 Current Status - FIXED AND RUNNING

### ✅ CRITICAL ISSUES RESOLVED
- **TypeError in joint angle limits calculation**: FIXED - Properly unpacking tuple return values
- **PyTorch module loading**: FIXED - Enhanced SLURM scripts with module purge and verification
- **All syntax errors**: RESOLVED - All evaluation scripts compile without errors

### Running SLURM Jobs (CORRECTED VERSIONS)
- **2 jobs currently running** on GPU nodes with fixes applied
- **2 jobs pending** (waiting for resources)
- **Total evaluation jobs**: 4 (restarted with fixes)

### Active Evaluations (FIXED VERSIONS)
| Job ID | Model Type | Eval Model | Status | Runtime | Node | Progress |
|--------|------------|------------|--------|---------|------|----------|
| 6304970 | raw | mixformer | RUNNING | 1:17 | str-gpu4 | Starting evaluation |
| 6304972 | transformer | mixformer | RUNNING | 1:17 | str-gpu18 | Starting evaluation |
| 6304974 | pmr | mixformer | PENDING | - | - | Waiting for resources |
| 6304976 | dmr | mixformer | PENDING | - | - | Waiting for resources |

### Previous Jobs (CANCELLED DUE TO BUGS)
| Job ID | Model Type | Status | Issue | Resolution |
|--------|------------|--------|-------|------------|
| 6304226 | transformer | CANCELLED | TypeError in joint angle limits | Fixed tuple unpacking |
| 6304227 | raw | CANCELLED | TypeError in joint angle limits | Fixed tuple unpacking |
| 6304230 | transformer | CANCELLED | TypeError in joint angle limits | Fixed tuple unpacking |
| 6304231 | raw | CANCELLED | TypeError in joint angle limits | Fixed tuple unpacking |

## 📁 Result File Locations

### 1. Comprehensive Evaluation Reports
**Location**: `results/comprehensive/`
- `evaluation_report_20250827_102303.md` - Latest comprehensive report
- `evaluation_report_20250827_102254.md` - Previous report
- `evaluation_report_20250827_102245.md` - Previous report
- `evaluation_report_20250827_102237.md` - Previous report
- **Total**: 10 comprehensive evaluation reports

### 2. MLM Evaluation Results
**Location**: `results/comprehensive_mlm_evaluation/`
- **JSON Results**: 18 files with detailed metrics
  - `ntu_cv_temporal_0.3_spatial_0.3_comprehensive.json`
  - `ntu_cv_temporal_0.3_spatial_0.5_comprehensive.json`
  - `ntu_cv_temporal_0.3_spatial_0.7_comprehensive.json`
  - `ntu_cv_temporal_0.5_spatial_0.3_comprehensive.json`
  - `ntu_cv_temporal_0.5_spatial_0.5_comprehensive.json`
  - And 13 more configuration combinations

### 3. Analysis and Visualization Results
**Location**: `results/analysis/plots/`
- `improvement_distribution.png` - Performance improvement distribution
- `improvement_heatmap.png` - Heatmap of improvements
- `raw_vs_mlm_comparison.png` - Raw vs MLM comparison
- `raw_vs_mlm_scatter.png` - Scatter plot comparison

**Location**: `results/comprehensive_mlm_evaluation/reports/`
- `ar_accuracy_heatmap_ntu_cv.png` - Action recognition accuracy heatmap
- **Total**: 56 visualization files

### 4. SLURM Job Logs
**Location**: `logs/`
- **Output Files (.out)**: 50 files containing job execution logs
- **Error Files (.err)**: 50 files containing progress and error information
- **Total**: 109 log files in 3 subdirectories

#### Current Job Logs (Active)
- `logs/comprehensive_eval_transformer_ntu_cv_6304226.out/.err`
- `logs/comprehensive_eval_raw_ntu_cv_6304227.out/.err`
- `logs/comprehensive_eval_transformer_ntu_cv_6304230.out/.err`
- `logs/comprehensive_eval_raw_ntu_cv_6304231.out/.err`
- `logs/eval_transformer_6304222.out/.err`

#### Completed Job Logs (Recent)
- `logs/comprehensive_eval_dmr_ntu_cv_6304229.out/.err` (Completed)
- `logs/comprehensive_eval_pmr_ntu_cv_6304228.out/.err` (Completed)

### 5. SLURM Scripts
**Location**: `slurm_out/`
- `comprehensive_eval_transformer_ntu_cv.sh`
- `comprehensive_eval_raw_ntu_cv.sh`
- `comprehensive_eval_pmr_ntu_cv.sh`
- `comprehensive_eval_dmr_ntu_cv.sh`

## 📊 Expected Results Structure

### When Jobs Complete, Results Will Be Available In:

#### Transformer Model Results
```
results/comprehensive/ntu_cv/transformer_mixformer/
├── metrics.json                 # Core performance metrics
├── ar_results.json             # Action recognition detailed results
├── ri_results.json             # Re-identification detailed results
├── gc_results.json             # Gender classification detailed results
├── physical_plausibility.json  # Physical metrics (5 required)
├── per_actor_breakdown.json    # Per-actor performance
├── per_action_breakdown.json   # Per-action performance
└── visualizations/             # Generated plots and charts
```

#### Raw Data Baseline Results
```
results/comprehensive/ntu_cv/raw_mixformer/
├── metrics.json                 # Baseline performance metrics
├── ar_results.json             # Action recognition baseline
├── ri_results.json             # Re-identification baseline
├── gc_results.json             # Gender classification baseline
├── physical_plausibility.json  # Physical metrics baseline
└── visualizations/             # Baseline visualizations
```

#### PMR/DMR Model Results
```
results/comprehensive/ntu_cv/pmr_mixformer/
results/comprehensive/ntu_cv/dmr_mixformer/
├── metrics.json                 # Anonymization model metrics
├── privacy_utility_tradeoff.json # Privacy vs utility analysis
├── ar_results.json             # Post-anonymization AR performance
├── ri_results.json             # Post-anonymization RI performance
├── gc_results.json             # Post-anonymization GC performance
├── physical_plausibility.json  # Physical realism after anonymization
└── visualizations/             # Anonymization visualizations
```

## 🔍 Key Metrics Being Evaluated

### Core Performance Metrics
1. **Action Recognition (AR) Accuracy**: Classification performance on 60 action classes
2. **Re-identification (RI) Accuracy**: Person identification performance
3. **Gender Classification (GC) Accuracy**: Gender prediction performance
4. **Mean Squared Error (MSE)**: Skeleton reconstruction quality

### Physical Plausibility Metrics (5 Required)
1. **Bone Length Consistency**: Maintains anatomical proportions
2. **Joint Angle Limits**: Respects human joint constraints
3. **Temporal Smoothness**: Ensures smooth motion transitions
4. **Velocity Consistency**: Realistic movement velocities
5. **Foot Contact Consistency**: Proper ground contact modeling

### Additional Analysis
- **Per-actor Performance**: Individual subject analysis
- **Per-action Performance**: Action-specific performance breakdown
- **Privacy-Utility Tradeoff**: For PMR/DMR models
- **Comparative Analysis**: Raw vs processed data comparison

## 📈 Progress Monitoring

### Real-time Monitoring Commands
```bash
# Check job status
squeue -u $USER

# Monitor specific job progress
tail -f logs/comprehensive_eval_*_6304226.err

# Track all evaluations
python scripts/track_evaluations.py --detailed

# Check results summary
python scripts/track_evaluations.py --results-only
```

### Progress Indicators
- **Raw Data Evaluation**: ~6s/batch, ~625 batches total (~62 minutes)
- **Transformer Evaluation**: ~9s/batch, ~625 batches total (~93 minutes)
- **PMR/DMR Evaluation**: Similar to transformer (~90-100 minutes)

## 🎯 Final Deliverables

### When All Jobs Complete, You Will Have:

1. **Comprehensive Evaluation Report** (`results/comprehensive/evaluation_report_FINAL.md`)
   - Executive summary of all model performance
   - Comparative analysis across all models
   - Recommendations and insights

2. **Detailed Metrics Files** (JSON format)
   - Quantitative results for all models
   - Statistical significance tests
   - Performance breakdowns

3. **Visualization Suite** (PNG/PDF files)
   - Performance comparison charts
   - Physical plausibility visualizations
   - Privacy-utility tradeoff plots

4. **Complete Log Archive** (`logs/`)
   - Full execution logs for reproducibility
   - Error logs for debugging
   - Performance profiling data

## 🚨 Monitoring Alerts

### Check These Locations for Issues:
- **Error Logs**: `logs/*_6304*.err` - Look for CUDA errors or crashes
- **Output Logs**: `logs/*_6304*.out` - Check for successful completion
- **Job Status**: `squeue -u $USER` - Monitor for failed jobs

### Expected Completion Times:
- **Job 6304226** (transformer): ~1.5 hours remaining
- **Job 6304227** (raw): ~1 hour remaining  
- **Job 6304230** (transformer): ~1.5 hours remaining
- **Job 6304231** (raw): ~1 hour remaining
- **Job 6304222** (transformer): Status unknown
- **Jobs 6304232/6304233** (PMR/DMR): Will start after current jobs complete

## 📞 Support Commands

```bash
# Get detailed job information
scontrol show job 6304226

# Cancel a job if needed
scancel 6304226

# Check node status
sinfo -N

# Monitor GPU usage
ssh str-gpu20.charlotte.edu nvidia-smi

# Generate current status report
python scripts/track_evaluations.py --detailed --hours 2
```

---

**Note**: This reference will be updated as evaluations complete. Check the tracking script for real-time status updates.
