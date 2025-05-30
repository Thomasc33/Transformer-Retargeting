# 🎮 Interactive Features Guide

## Overview

The Transformer Retargeting system now includes comprehensive interactive features that make it incredibly easy to use. The system intelligently handles model dependencies, prompts for overwriting existing results, and guides users through complex workflows.

## 🚀 **Interactive Pipeline (`scripts/pipeline.py --interactive`)**

### Features

1. **Smart Step Overwriting**
   - Detects completed steps automatically
   - Prompts user whether to re-run or skip
   - Shows existing file sizes and locations

2. **Intelligent Evaluation Selection**
   - Lists available evaluations with descriptions
   - Checks model dependencies automatically
   - Offers to train missing models

3. **Model Dependency Management**
   - Detects missing SGN/MixFormer/Transformer models
   - Automatically adds training steps if needed
   - Prevents evaluation failures due to missing models

### Example Workflow

```bash
python scripts/pipeline.py --interactive
```

**Interactive Flow:**
1. **Dataset Selection** - Choose from NTU, NTU120, ETRI
2. **Setting Selection** - Cross-subject (cs) or Cross-view (cv)
3. **Step Selection** - Choose pipeline steps or use presets
4. **Evaluation Selection** - Pick specific evaluations to run
5. **Model Dependency Check** - System checks for required models
6. **Missing Model Training** - Offers to train missing models
7. **Execution Mode** - Direct, SLURM, or Windows batch generation

## 🧪 **Interactive Evaluation (`scripts/evaluate.py --interactive`)**

### Features

1. **Custom Evaluation Selection**
   - Choose specific evaluations from a comprehensive list
   - Model dependency checking for each evaluation
   - Automatic model training prompts

2. **Experiment Set Management**
   - Pre-configured experiment sets (critical, quick, complete)
   - Dependency checking for entire sets
   - Time estimation and progress tracking

3. **Single Experiment Execution**
   - Browse all available experiments
   - Check completion status
   - Model dependency validation

### Available Evaluations

- **Transformer Autoencoder** - Main model evaluation
- **SGN Action Recognition** - SGN model for action recognition
- **SGN Re-Identification** - SGN model for person re-identification
- **SGN Gesture Classification** - SGN model for gesture classification
- **MixFormer Action Recognition** - MixFormer for action recognition
- **MixFormer Re-Identification** - MixFormer for person re-identification
- **MixFormer Gesture Classification** - MixFormer for gesture classification
- **Privacy Utility Analysis** - Privacy-preserving evaluation
- **Loss Component Ablation** - Loss function analysis
- **Masking Ratio Analysis** - MLM masking strategy evaluation
- **Comprehensive Evaluation** - Full evaluation suite

### Example Usage

```bash
python scripts/evaluate.py --interactive
```

**Menu Options:**
1. **📋 List all experiments** - Browse available experiments
2. **🎯 Run experiment set** - Execute pre-configured sets
3. **🧪 Run single experiment** - Run individual experiments
4. **🔧 Run custom evaluation selection** - Choose specific evaluations
5. **⚙️ Configure experiment** - Modify experiment parameters
6. **🚪 Exit** - Exit the interactive mode

## 🔄 **Smart Step Skipping**

### How It Works

The system automatically detects completed steps by checking for:

- **Preprocessing** - Processed data files (`.pkl`, `.pt`)
- **Sampling** - Paired sample files (`*_paired_*.pt`)
- **Pretraining** - Pretrained model files (`.pth`)
- **Training** - Trained model checkpoints
- **Evaluation** - Evaluation result files (`.json`)

### Interactive Prompts

When a completed step is detected:

```
⚠️  Step 'preprocess' appears to be already completed.
📁 Existing files:
   • data/ntu/ntu_cv_processed.pkl (45.2 MB)

🔄 Do you want to re-run 'preprocess' and overwrite existing results? (y/n):
```

**Options:**
- **Yes (y)** - Re-run the step and overwrite existing results
- **No (n)** - Skip the step and continue with existing results

## 🏗️ **Model Dependency Management**

### Automatic Detection

The system checks for required models based on selected evaluations:

```
⚠️  Some evaluations require models that don't exist yet:

🏗️  Missing SGN models:
   • ar_ntu_cv
   • ri_ntu120_cs

🏗️  Missing MIXFORMER models:
   • gc_etri_cv

🏋️  Do you want to automatically train the missing models first? (y/n):
```

### Model Types Checked

1. **SGN Models**
   - Action Recognition (AR)
   - Re-Identification (RI)
   - Gesture Classification (GC)

2. **MixFormer Models**
   - Action Recognition (AR)
   - Re-Identification (RI)
   - Gesture Classification (GC)

3. **Transformer Models**
   - Main autoencoder model
   - Pretrained encoders

### Training Integration

If missing models are detected:
- **Automatic Training** - Adds training steps to pipeline
- **Skip Missing** - Continues with available models only
- **Manual Training** - User can train models separately

## 🎯 **Evaluation Sets**

### Pre-configured Sets

1. **Critical** - Essential evaluations (~17 hours)
   - Transformer autoencoder
   - SGN action recognition
   - MixFormer action recognition
   - Privacy utility analysis

2. **Quick** - Fast testing (~9 hours)
   - Basic functionality tests
   - Core model evaluations

3. **Complete** - All evaluations (~85 hours)
   - Comprehensive evaluation suite
   - All model variants
   - Full ablation studies

4. **Paper Ready** - Publication-ready (~50 hours)
   - Key results for papers
   - Statistical significance tests
   - Visualization generation

## 💡 **Best Practices**

### For First-Time Users

1. **Start Interactive** - Always use `--interactive` mode first
2. **Check Data** - Run `python scripts/validate_data_paths.py` first
3. **Use Critical Set** - Start with critical evaluations
4. **Monitor Dependencies** - Let system handle model training

### For Advanced Users

1. **Custom Evaluations** - Use custom evaluation selection
2. **HPC Generation** - Use `--slurm` for cluster execution
3. **Resume Capability** - Use `--resume-from` for interrupted runs
4. **Batch Processing** - Generate scripts for multiple configurations

### For Researchers

1. **Experiment Sets** - Use pre-configured sets for consistency
2. **Model Validation** - Always check model dependencies
3. **Result Tracking** - Monitor completion status
4. **Reproducibility** - Use configuration files for consistency

## 🔧 **Troubleshooting**

### Common Issues

1. **Missing Models**
   - **Solution**: Let system train missing models automatically
   - **Alternative**: Train models manually with specific scripts

2. **Data Not Found**
   - **Solution**: Run `python scripts/validate_data_paths.py`
   - **Check**: Ensure correct data directory structure

3. **Step Already Completed**
   - **Solution**: Choose to overwrite or skip in interactive prompt
   - **Alternative**: Use `--resume-from` to start from specific step

4. **Evaluation Failures**
   - **Solution**: Check model dependencies first
   - **Debug**: Use single experiment mode to isolate issues

### Getting Help

```bash
# Check system status
python scripts/setup.py --check-only

# Validate data structure
python scripts/validate_data_paths.py

# List available experiments
python scripts/evaluate.py --list

# Interactive help
python scripts/pipeline.py --interactive
python scripts/evaluate.py --interactive
```

## 🎉 **Summary**

The interactive features make the Transformer Retargeting system:

- **🎮 User-Friendly** - Guided workflows for all skill levels
- **🧠 Intelligent** - Automatic dependency management
- **🔄 Efficient** - Smart step skipping and resumption
- **🎯 Flexible** - Custom evaluation selection
- **🏗️ Robust** - Model validation and training integration
- **📊 Comprehensive** - Full evaluation suite management

**Start exploring:** `python scripts/pipeline.py --interactive`
