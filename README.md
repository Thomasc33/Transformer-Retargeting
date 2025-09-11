# Transformer Retargeting

Clean, unified training + evaluation with one entrypoint and an auto-updating dashboard.

- Run everything from tmr.py (interactive or CLI)
- Results collect under results/ and are summarized in results.html at repo root
- Slurm evaluation logs live in logs/slurm; models in data/models*, data under data/
- GIFs/images/videos are kept under version control for reports

## Repository layout (target)
- data/           datasets, models, and outputs
  - models/       saved/trained models (migrated from trained_models/)
  - models_output/ training outputs and checkpoints (migrated from output/)
- configs/        YAML and experiment configs
- docs/           documentation and scripts (migrated legacy: experiments/, bash/)
  - scripts/      CLI helpers and legacy bash tools
  - experiments/  documentation of experiment setups
- eval/           unified evaluation package (python -m eval)
- logs/           logs and Slurm outputs
  - slurm/
- src/            library code (training, data, utils)

Note: legacy directories were migrated automatically via `python tmr.py refactor-structure`.

## Quick start

Interactive menu

```
python tmr.py
```

Common commands

```
# Evaluate critical set and refresh dashboard
python tmr.py eval --set critical
python tmr.py dash

# Clean evaluation outputs (keeps models)
python tmr.py clean-results

# Migrate saved models to data/models*
python tmr.py migrate-models

# Move legacy dirs into the target layout
python tmr.py refactor-structure
```

## Cross-platform shortcuts
- macOS/Linux: `make eval-critical`, `make dash`, `make clean-results`, `make validate`
- Windows: `eval-critical.cmd`, `dash.cmd`

## Dashboard
- Open `results.html` at the repo root
- Missing artifacts/metrics are called out so the dashboard is never empty

## Unified entrypoint: tmr.py

- Interactive mode (no args):

```bash
python tmr.py
```

- Evaluate (critical set) and rebuild dashboard:

```bash
python tmr.py eval --set critical
python tmr.py dash
```

- Clean old evaluation results (keeps training/saved models):

```bash
python tmr.py clean-results
```

- Validate environment (torch, CUDA, data paths):

```bash
python tmr.py validate
```

- Windows shortcuts:
  - eval-critical.cmd
  - dash.cmd

See docs/EVALUATION.md and docs/STRUCTURE.md for details.

## 📁 Data Structure (IMPORTANT!)

**NTU120 uses BOTH NTU60 and NTU120 skeleton data!**

```
📁 data/
├── 📁 nturgbd_raw/                    # Main raw data directory
│   ├── 📁 nturgb+d_skeletons/         # NTU RGB+D 60 skeleton files
│   └── 📁 nturgb+d_skeletons120/      # NTU RGB+D 120 skeleton files
├── 📁 etri_raw/                       # ETRI raw data (if using)
├── 📁 ntu/                           # Processed NTU60 data
├── 📁 ntu120/                        # Processed NTU120 data
└── 📁 etri/                          # Processed ETRI data
```

**Validate your data structure:**
```bash
python scripts/validate_data_paths.py
```

## 🎮 New Interactive Features

### 🧠 **Intelligent Model Management**
The system automatically detects missing models and offers to train them:

```bash
python scripts/evaluate.py --interactive
# System detects: "SGN model for action recognition not found"
# Prompts: "Do you want to train the missing SGN model? (y/n)"
```

### 🔄 **Smart Overwrite Prompts**
When steps are already completed, the system asks before overwriting:

```bash
python scripts/pipeline.py --interactive
# System detects: "Preprocessing already completed (45.2 MB)"
# Prompts: "Do you want to re-run preprocessing and overwrite? (y/n)"
```

### 🎯 **Custom Evaluation Selection**
Choose exactly which evaluations to run:

```bash
python scripts/evaluate.py --interactive
# Choose from: Transformer, SGN AR/RI/GC, MixFormer AR/RI/GC, etc.
# System automatically checks and trains required models
```

**[📚 Complete Interactive Features Guide](docs/INTERACTIVE_FEATURES.md)**

## ⚡ Smart Step Skipping

The system automatically detects completed steps and skips them:

- **Preprocessing** - Checks for processed data files
- **Sampling** - Checks for paired sample files
- **Pretraining** - Checks for pretrained model files
- **Training** - Checks for trained model checkpoints
- **Evaluation** - Checks for evaluation result files

```bash
# Pipeline automatically skips completed steps
python scripts/pipeline.py --quick-start --dataset ntu --setting cv

# Force re-run specific step
python scripts/preprocess.py --dataset ntu --setting cv  # Will skip if already done
```

## 🎮 Interactive Mode Examples

```bash
# Interactive evaluation (experiment selection)
python scripts/evaluate.py --interactive

# Interactive training configuration
python scripts/train.py --interactive

# Interactive pretraining setup
python scripts/pretrain.py --interactive

# Interactive data preprocessing
python scripts/preprocess.py --interactive
```

## 🖥️ HPC & Cross-Platform Support

```bash
# Generate SLURM jobs for HPC
python scripts/train.py --model transformer --dataset ntu --setting cv --slurm
python scripts/pipeline.py --quick-start --dataset ntu --setting cv --slurm

# Generate Windows batch files
python scripts/train.py --model sgn --dataset ntu120 --setting cs --windows
python scripts/pipeline.py --quick-start --dataset ntu --setting cv --windows
```

## 📚 Documentation

- **[Complete Documentation](docs/README.md)** - Detailed guides and examples
- **[Consolidation Summary](CONSOLIDATION_SUMMARY.md)** - What changed and why
- **[Data Validation](scripts/validate_data_paths.py)** - Check your data structure

## Abstract

Motion retargeting is a critical task in computer animation, virtual reality, and motion analysis. Our transformer-based approach addresses this challenge by learning to map motion from one skeleton to another while maintaining the essential action characteristics. We utilize a spatial-temporal encoder to extract features from input skeletons and an autoregressive decoder to generate retargeted motion sequences. This approach enables effective motion transfer between different body proportions while preserving the semantic meaning of actions.

## Architecture

Our model consists of two main components:
1. **Spatial-Temporal Encoder**: Adapted from the Skeleton-MixFormer architecture to extract rich representations of skeleton motion
2. **Autoregressive Decoder**: A transformer decoder that generates the retargeted motion sequence frame by frame

<p align="center">
  <img src="path/to/architecture_diagram.png" alt="Architecture Diagram" width="600"/>
</p>

## Dependencies

+ Python >= 3.6
+ PyTorch >= 1.1.0
+ tqdm, tensorboardX
+ Optionally: CUDA-capable GPU for faster training and inference

## Data Preparation

Our model can be trained on several skeleton datasets:

### Download Datasets

**Supported datasets:**
+ NTU RGB+D 60 Skeleton
+ NTU RGB+D 120 Skeleton
+ ETRI Human Action Recognition

#### NTU RGB+D 60 and 120

1. Request dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
    i. ```nturgbd_skeletons_s001_to_s017.zip``` (NTU RGB+D 60)
    ii. ```nturgbd_skeletons_s018_to_s032.zip``` (NTU RGB+D 120)
    iii. Extract above files to ```./data/nturgbd_raw```

#### Directory Structure

Put downloaded data into the following directory structure:
```
- data/
  - UAV-Human/
    - Skeleton
      ... # raw data of UAV-Human
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

+ Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:
```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```


# Training & Testing
### Training
+ Change the config file depending on what you want.
```
    # Example: training SKMIXF on NTU RGB+D cross subject with GPU 0
    python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/skmixf --device 0
    # Example: training provided baseline on NTU RGB+D cross subject
    python main.py --config config/nturgbd-cross-subject/default.yaml --model model.baseline.Model--work-dir work_dir/ntu/csub/baseline --device 0
```
+ To train model on NTU RGB+D 60/120 with bone or motion modalities, setting ```bone``` or ```vel``` arguments in the config file ```default.yaml``` or in the command line.
```
    # Example: training SKMIXF on NTU RGB+D 120 cross subject under bone modality
    python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --work-     dir work_dir/ntu120/csub/skmixf_bone --device 0
```
+ To train model on NW-UCLA with bone or motion modalities, you need to modify ```data_path``` in ```train_feeder_args``` and ```test_feeder_args``` to "bone" or "motion" or "bone motion", and run
```
    python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/skmixf_xxx --device 0
```
+ To train model on UAV-Human with bone or motion modalities, you need to modify ```data_path``` in ```train_feeder_args``` and ```test_feeder_args``` to "bone" or "motion" or "bone motion", and run
```
    python main.py --config config/uav/default.yaml --work-dir work_dir/uav/skmixf_xxx --device 0
```

### Testing

+ To test the trained models saved in <work_dir>, run the following command:

```
    python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --         device 0
```

+ To ensemble the results of different modalities, run

```
    # Example: ensemble four modalities of SkMIXF on NTU RGB+D cross subject
    python ensemble.py --dataset ntu/xsub  --joint-dir  work_dir/ntu/csub/skmixf --bone-dir  work_dir/ntu/csub/skmixf_bone --joint-motion-dir  work_dir/ntu120/csub/skmixf_motion  --bone-motion-dir work_dir/ntu/csub/skmixf_bone_motion  --joint-k2-dir work_dir/ntu120/csub/skmixf_joint_k2  --joint-motion-k2-dir  work_dir/ntu120/csub/skmixf_joint_motion_k2
```

### Pretrained model
+ Pretrained weights for NTU RGB+D 60 and 120 can be downloaded from the following link [[Google Drive]](https://drive.google.com/file/d/15Ahneq5_IgurficrYb3PiiLeEFyS8lBQ/view?usp=share_link)

## Acknowledgements
This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) and [Info-GCN](https://github.com/stnoah1/infogcn) The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!


