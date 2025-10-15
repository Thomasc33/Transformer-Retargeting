# Transformer Motion Retargeting (TMR)

> **Privacy-Preserving Motion Retargeting with Transformer Architecture**

A comprehensive system for skeleton-based motion retargeting that preserves action semantics while anonymizing identity. Built with a production-ready CLI interface, automated evaluation pipelines, and interactive dashboards.

---

## 🚀 Quick Start

**Interactive Mode** (Recommended for first-time users):
```bash
python tmr.py
```

**Check System Status**:
```bash
python tmr.py  # Select option 9: Repository Status
```

**Run Evaluation**:
```bash
python src/evaluation/eval_anonymization_v2.py --only-tmr --dataset ntu --setting cv --test-samples 100
```

**View Results Dashboard**:
```bash
open results.html  # macOS
firefox results.html  # Linux
```

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [tmr.py - Main Interface](#tmrpy---main-interface)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results & Visualization](#results--visualization)
- [SLURM Integration](#slurm-integration)
- [Known Issues](#known-issues)
- [Documentation](#documentation)

---

## 🎯 Overview

**Transformer Motion Retargeting (TMR)** is a deep learning system that transfers motion from one person to another while:
- ✅ **Preserving Action**: The retargeted motion maintains the original action semantics
- ✅ **Anonymizing Identity**: The retargeted motion appears to be performed by a different person
- ✅ **Maintaining Quality**: The retargeted motion is physically plausible and natural

### Key Features

- 🎮 **Interactive CLI**: Comprehensive menu-driven interface via `tmr.py`
- 📊 **Auto-Updating Dashboard**: Real-time results visualization in `results.html`
- 🔄 **Complete Pipeline**: Data → Pretrain → Train → Evaluate → Visualize
- 🖥️ **SLURM Integration**: Automated job submission and tracking for HPC clusters
- 📈 **Experiment Tracking**: Track all experiments, models, and results
- 🧪 **Comprehensive Evaluation**: AR, RI, GC, physical plausibility, robustness
- 🎨 **Modern UI**: Beautiful dark-themed dashboard with interactive charts

---

## 🏗️ Architecture

TMR uses a **two-stage architecture**:

### Stage 1: Spatial-Temporal Encoder
- **Base**: Pretrained Skeleton-MixFormer
- **Input**: Skeleton sequences (N, C, T, V, M) = (batch, 3, 64, 25, 1)
- **Output**: Rich spatial-temporal features
- **Purpose**: Extract action and identity features from skeleton motion

### Stage 2: Autoregressive Decoder
- **Type**: Transformer decoder with causal masking
- **Input**: Encoder features + target skeleton reference
- **Output**: Retargeted skeleton sequence (frame-by-frame)
- **Purpose**: Generate motion that preserves action but changes identity

### Current Status

⚠️ **Known Issue**: The current TMR model is fundamentally broken (0% AR accuracy). See [RETRAINING_PLAN.md](RETRAINING_PLAN.md) for details and solution.

**Working Components**:
- ✅ Data loading and preprocessing
- ✅ Baseline models (Mixformer: 74% AR, SGN: 90% AR)
- ✅ Evaluation pipeline
- ✅ Interactive CLI and dashboard

**Needs Retraining**:
- ❌ TMR model (currently outputs identity function or corrupted skeletons)

---

## 📦 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0 (for GPU support)

### Setup

```bash
# Clone repository
git clone git@github.com:Thomasc33/Transformer-Retargeting.git
cd Transformer-Retargeting

# Install dependencies
pip install -r requirements.txt

# Verify installation
python tmr.py  # Select option 0: Validate Environment
```

---

## 📁 Repository Structure

```
Transformer-Retargeting/
├── tmr.py                    # 🎯 MAIN ENTRY POINT - Interactive CLI
├── results.html              # 📊 Interactive results dashboard
├── results.json              # 📈 Results data (auto-generated)
├── Makefile                  # 🛠️ Cross-platform shortcuts
│
├── data/                     # 💾 Datasets and models
│   ├── nturgbd_raw/          # Raw NTU RGB+D data
│   │   ├── nturgb+d_skeletons/      # NTU 60
│   │   └── nturgb+d_skeletons120/   # NTU 120
│   ├── ntu_cv_paired_comprehensive.pt
│   ├── ntu_cs_paired_10000_2000.pt
│   └── ntu120_cv_paired_10000_2000.pt
│
├── models/                   # 🤖 Trained models
│   ├── tmr/                  # TMR models (needs retraining)
│   └── baselines/            # Baseline models (working)
│
├── src/                      # 📚 Source code
│   ├── cli/                  # CLI modules (10 modules, 2500+ lines)
│   │   ├── menu.py           # Main interactive menu
│   │   ├── data_commands.py  # Data management
│   │   ├── train_commands.py # Training operations
│   │   ├── eval_commands.py  # Evaluation operations
│   │   ├── experiment.py     # Experiment tracking
│   │   ├── pipeline.py       # Pipeline orchestration
│   │   ├── repo_manager.py   # Repository status
│   │   ├── slurm_manager.py  # SLURM job management
│   │   └── utils.py          # Utilities
│   ├── data/                 # Data loading and processing
│   ├── models/               # Model architectures
│   ├── training/             # Training scripts
│   └── evaluation/           # Evaluation scripts
│       ├── eval_anonymization_v2.py  # Main evaluation script
│       └── eval_model_main.py        # Evaluation utilities
│
├── scripts/                  # 🔧 Utility scripts
│   ├── eval_same_action.py   # Same-action evaluation
│   └── quick_test_tmr.py     # Quick system test
│
├── eval/                     # 🧪 Evaluation suite
├── configs/                  # ⚙️ Configuration files
├── logs/                     # 📝 Training and evaluation logs
│   └── slurm/                # SLURM job logs
├── results/                  # 📊 Evaluation results
└── docs/                     # 📖 Documentation
    ├── RETRAINING_PLAN.md    # Detailed retraining guide
    ├── STATUS_REPORT.md      # Current status and findings
    └── OPTIONS_MOVING_FORWARD.md  # Solution approaches
```

---

## 🎮 tmr.py - Main Interface

`tmr.py` is your **single entry point** for all operations. It provides both an interactive menu and command-line interface.

### Interactive Mode (Recommended)

```bash
python tmr.py
```

**Menu Options**:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     TMR - Transformer Motion Retargeting                     ║
║                          Interactive Command Center                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ 0. Validate Environment          - Check PyTorch, CUDA, dependencies        │
│ 1. Data Operations                - List, validate, preprocess datasets      │
│ 2. Training Operations            - Train TMR, pretrain encoder              │
│ 3. Evaluation Operations          - Run evaluations, compare models          │
│ 4. Experiment Management          - Track experiments, view history          │
│ 5. Pipeline Operations            - Run complete workflows                   │
│ 6. SLURM Job Management           - Submit, monitor, cancel jobs             │
│ 7. Results & Visualization        - Generate dashboard, export results       │
│ 8. Utilities                      - Clean, migrate, backup                   │
│ 9. Repository Status              - View current state                       │
│ 10. Exit                                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Command-Line Interface

**Data Operations**:
```bash
python tmr.py data list              # List available datasets
python tmr.py data validate          # Validate data paths
python tmr.py data preprocess        # Preprocess raw data
```

**Training Operations**:
```bash
python tmr.py train tmr              # Train TMR model
python tmr.py train pretrain         # Pretrain encoder
python tmr.py train baseline         # Train baseline models
```

**Evaluation Operations**:
```bash
python tmr.py eval run               # Run evaluation
python tmr.py eval compare           # Compare models
python tmr.py eval analyze           # Analyze results
```

**Experiment Management**:
```bash
python tmr.py exp create             # Create new experiment
python tmr.py exp list               # List experiments
python tmr.py exp status             # Check experiment status
```

**Pipeline Operations**:
```bash
python tmr.py pipeline full          # Run complete pipeline
python tmr.py pipeline quick         # Quick evaluation + dashboard
```

**SLURM Operations**:
```bash
python tmr.py slurm submit           # Submit job
python tmr.py slurm status           # Check job status
python tmr.py slurm cancel           # Cancel job
```

**Results & Visualization**:
```bash
python tmr.py viz dashboard          # Generate dashboard
python tmr.py viz export             # Export results
```

**Utilities**:
```bash
python tmr.py util clean             # Clean old results
python tmr.py util migrate           # Migrate models
python tmr.py util backup            # Backup important files
```

**Repository Status**:
```bash
python tmr.py status                 # Show repository status
```

---

## 💾 Data Preparation

### Supported Datasets

- **NTU RGB+D 60**: 56,880 samples, 60 action classes, 40 subjects
- **NTU RGB+D 120**: 114,480 samples, 120 action classes, 106 subjects
- **ETRI**: Korean action recognition dataset

### Download NTU RGB+D

1. **Request Access**: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. **Download Skeleton Data**:
   - `nturgbd_skeletons_s001_to_s017.zip` (NTU 60)
   - `nturgbd_skeletons_s018_to_s032.zip` (NTU 120)
3. **Extract to**:
   ```
   data/nturgbd_raw/nturgb+d_skeletons/      # NTU 60
   data/nturgbd_raw/nturgb+d_skeletons120/   # NTU 120
   ```

### Verify Data Structure

```bash
python tmr.py data validate
```

**Expected Structure**:
```
data/
├── nturgbd_raw/
│   ├── nturgb+d_skeletons/          # NTU 60 raw files
│   │   ├── S001C001P001R001A001.skeleton
│   │   ├── S001C001P001R001A002.skeleton
│   │   └── ... (56,880 files)
│   └── nturgb+d_skeletons120/       # NTU 120 raw files
│       ├── S018C001P008R001A001.skeleton
│       └── ... (57,600 additional files)
├── ntu_cv_paired_comprehensive.pt   # Processed (auto-generated)
├── ntu_cs_paired_10000_2000.pt      # Processed (auto-generated)
└── ntu120_cv_paired_10000_2000.pt   # Processed (auto-generated)
```

### Preprocess Data

```bash
python tmr.py data preprocess
# Or use the interactive menu: Option 1 → Preprocess Data
```

**What it does**:
1. Loads raw skeleton files
2. Removes bad/corrupted samples
3. Normalizes skeleton coordinates
4. Creates paired samples for retargeting
5. Saves processed data as `.pt` files

---

## 🎓 Training

### Stage 1: Pretrain Encoder (Optional but Recommended)

```bash
python tmr.py train pretrain
```

**Purpose**: Pretrain the Skeleton-MixFormer encoder on action recognition
**Duration**: ~12 hours on 4x V100 GPUs
**Output**: `models/pretrained/encoder.pth`

### Stage 2: Train TMR

```bash
python tmr.py train tmr
```

**Purpose**: Train the full TMR model (encoder + decoder)
**Duration**: ~24 hours on 4x V100 GPUs
**Output**: `models/tmr/model.pth`

### Training Configuration

Edit `configs/train_config.yaml`:
```yaml
# Model
model:
  encoder: 'skeleton_mixformer'
  decoder: 'autoregressive_transformer'
  hidden_dim: 256
  num_layers: 6
  num_heads: 8

# Training
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: 'adam'
  scheduler: 'cosine'

# Loss weights
loss:
  reconstruction: 1.0
  action_recognition: 0.5  # Cooperative AR loss
  identity_confusion: 0.3  # Privacy loss
```

### SLURM Training

```bash
python tmr.py slurm submit --job train_tmr --gpus 4 --time 24:00:00
```

---

## 🧪 Evaluation

### Quick Evaluation

```bash
python src/evaluation/eval_anonymization_v2.py \
  --only-tmr \
  --dataset ntu \
  --setting cv \
  --test-samples 100 \
  --device cuda
```

### Full Evaluation

```bash
python src/evaluation/eval_anonymization_v2.py \
  --dataset ntu \
  --setting cv \
  --device cuda
```

**Metrics**:
- **AR (Action Recognition)**: Does retargeted motion preserve action? (Target: >70%)
- **RI (Re-Identification)**: Does retargeted motion confuse identity? (Target: >80%)
- **GC (Gait Cycle)**: Is motion physically plausible? (Target: >0.8)
- **MSE**: Reconstruction error (Lower is better)

### Same-Action Evaluation (New!)

Test TMR on pairs where both actors perform the **same action**:

```bash
python scripts/eval_same_action.py \
  --dataset ntu_cv \
  --num_pairs 100 \
  --device cuda
```

**Why?** This is a more controlled test since the action is consistent across both inputs. TMR should theoretically preserve action better in this scenario.

### Evaluation Results

Results are automatically saved to:
- `results/evaluation_results.json` - Raw data
- `results.html` - Interactive dashboard

---

## 📊 Results & Visualization

### Interactive Dashboard

```bash
open results.html  # macOS
firefox results.html  # Linux
start results.html  # Windows
```

**Features**:
- 📈 **6 Interactive Tabs**: Overview, Models, Experiments, Visualizations, Comparison, README
- 📊 **3 Chart.js Charts**: Overview, AR Comparison, Privacy Comparison
- 🎨 **39 Card Components**: Model cards, experiment cards, metric cards
- 📥 **Export Functionality**: Download results as JSON
- 📱 **Responsive Design**: Works on desktop, tablet, mobile
- 🌙 **Dark Theme**: Modern glassmorphism design

### Generate Dashboard

```bash
python tmr.py viz dashboard
```

### Export Results

```bash
python tmr.py viz export --format json
python tmr.py viz export --format csv
```

---

## 🖥️ SLURM Integration

### Submit Training Job

```bash
python tmr.py slurm submit \
  --job train_tmr \
  --gpus 4 \
  --time 24:00:00 \
  --mem 64G \
  --partition gpu
```

### Monitor Jobs

```bash
python tmr.py slurm status        # Check all jobs
python tmr.py slurm logs <job_id> # View job logs
```

### Cancel Jobs

```bash
python tmr.py slurm cancel <job_id>
python tmr.py slurm cancel --all
```

### Job Tracking

All submitted jobs are tracked in `jobs.json`:
```json
{
  "jobs": [
    {
      "id": "12345",
      "name": "train_tmr",
      "status": "running",
      "submitted": "2024-10-15 10:00:00",
      "gpus": 4,
      "time": "24:00:00"
    }
  ]
}
```

---

## ⚠️ Known Issues

### TMR Model is Broken (0% AR Accuracy)

**Problem**: All three TMR checkpoints are fundamentally broken:
- `model_all.pth` & `model.pth`: Output identity function (copying input)
- `model_baseline.pth`: Output corrupted/invalid skeletons

**Root Cause**: Missing cooperative AR loss during training

**Solution**: See [RETRAINING_PLAN.md](RETRAINING_PLAN.md) for detailed retraining guide

**Workaround**: Use baseline models (Mixformer, SGN) for now

### Baseline Models Work Correctly

✅ **Mixformer**: 74% AR, 86% RI
✅ **SGN**: 90% AR, 82% RI
✅ **Data Loading**: Working correctly
✅ **Evaluation Pipeline**: Working correctly

---

## 📚 Documentation

### Core Documentation

- **[RETRAINING_PLAN.md](RETRAINING_PLAN.md)** - Detailed guide for retraining TMR (26 KB, 600+ lines)
- **[STATUS_REPORT.md](STATUS_REPORT.md)** - Current status and findings (16 KB, 400+ lines)
- **[OPTIONS_MOVING_FORWARD.md](OPTIONS_MOVING_FORWARD.md)** - Analysis of 5 solution approaches
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference card for common tasks

### Additional Resources

- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Setup verification guide
- **[FIXES_COMPLETE.md](FIXES_COMPLETE.md)** - Summary of recent fixes
- **[BEFORE_AFTER_COMPARISON.md](BEFORE_AFTER_COMPARISON.md)** - Before/after comparison

---

## 🧪 Testing

### Quick System Test

```bash
python scripts/quick_test_tmr.py
```

**Tests**:
- ✅ Dataset detection (3/3 datasets)
- ✅ Data loading (1.6M train, 825K test)
- ✅ Model loading (TMR + baselines)
- ✅ Evaluation pipeline

### Same-Action Evaluation

```bash
python scripts/eval_same_action.py --dataset ntu_cv --num_pairs 100 --device cuda
```

**Purpose**: Test TMR on pairs where both actors perform the **same action**. This is a more controlled test since the action is consistent.

**Expected Results**:
- AR > 50%: TMR preserving action ✅
- RI > 80%: TMR achieving privacy ✅

---

## 🎯 Next Steps

### Immediate Actions

1. **Verify Setup**:
   ```bash
   python tmr.py  # Select option 0: Validate Environment
   python tmr.py  # Select option 9: Repository Status
   ```

2. **Test Evaluation Pipeline**:
   ```bash
   python src/evaluation/eval_anonymization_v2.py --only-tmr --dataset ntu --setting cv --test-samples 100
   ```

3. **Run Same-Action Experiment**:
   ```bash
   python scripts/eval_same_action.py --dataset ntu_cv --num_pairs 100 --device cuda
   ```

4. **View Results**:
   ```bash
   open results.html
   ```

### Long-Term Goals

1. **Retrain TMR** (See [RETRAINING_PLAN.md](RETRAINING_PLAN.md)):
   - Implement cooperative AR loss
   - Train Stage 1 (encoder with AR loss)
   - Train Stage 2 (full model with all losses)
   - Validate on test set

2. **Expand Evaluation**:
   - Physical plausibility metrics
   - Robustness analysis
   - Cross-dataset evaluation

3. **Optimize Performance**:
   - Model compression
   - Inference speed optimization
   - Memory efficiency

---

## 📊 Dataset Statistics

### NTU RGB+D 60
- **Samples**: 56,880
- **Actions**: 60 classes
- **Subjects**: 40 people
- **Views**: 3 camera angles
- **Splits**: Cross-Subject (CS), Cross-View (CV)

### NTU RGB+D 120
- **Samples**: 114,480
- **Actions**: 120 classes
- **Subjects**: 106 people
- **Views**: 3 camera angles
- **Splits**: Cross-Subject (CS), Cross-Setup (CSet)

### Processed Data
- **Training**: ~1.6M paired samples
- **Testing**: ~825K paired samples
- **Total**: ~2.4M paired samples

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Data Processing**: Borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch)
- **Skeleton-MixFormer**: Base encoder architecture
- **NTU RGB+D Dataset**: Provided by Nanyang Technological University

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review [RETRAINING_PLAN.md](RETRAINING_PLAN.md) for TMR-specific questions

---

**Last Updated**: October 15, 2024
**Version**: 2.0
**Status**: Production-Ready CLI, TMR Model Needs Retraining
