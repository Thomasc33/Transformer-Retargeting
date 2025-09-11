# Standalone Evaluation Script

The `standalone_eval.py` script provides a comprehensive, user-friendly way to evaluate and visualize trained models independently of the main pipeline system.

## Features

- **Multiple Model Types**: Support for transformer, PMR, DMR, raw data, and "all" models evaluation
- **SLURM Integration**: Submit evaluation jobs to SLURM with proper GPU partition configuration
- **Auto Model Discovery**: Automatically find default models or evaluate all available models
- **Validation Set Support**: Use validation set from comprehensive data based on camera
- **Flexible Evaluation**: Use SGN or MixFormer models for AR/RI/GC tasks
- **Rich Visualizations**: Create skeleton animations, comparisons, motion trails, and more
- **Interactive Mode**: User-friendly interface with path validation and model search
- **Command Line Mode**: Scriptable interface for automation
- **Comprehensive Reporting**: Detailed markdown reports with metrics, model summaries, and default paths
- **Improved Naming**: Clean, readable model names and organized result structure

## Quick Start

### Interactive Mode (Recommended for first-time users)

```bash
python scripts/standalone_eval.py --interactive
```

This will guide you through:
1. Model selection (with search functionality)
2. Configuration options
3. Evaluation and visualization choices

### Command Line Mode

```bash
# Basic evaluation
python scripts/standalone_eval.py --model-path model.pth --model-type transformer --dataset ntu --setting cv

# With visualization
python scripts/standalone_eval.py --model-path model.pth --model-type transformer --dataset ntu --setting cv --visualize --viz-types skeleton_animations

# Skip evaluation, only visualize
python scripts/standalone_eval.py --model-path model.pth --model-type transformer --dataset ntu --setting cv --no-eval --visualize
```

## Command Line Options

### Required (for command line mode)
- `--model-path`: Path to the model file (use "auto" to find default model)
- `--model-type`: Type of model (transformer, pmr, dmr, raw, all)

### Dataset Configuration
- `--dataset`: Dataset to use (ntu, ntu120, etri) [default: ntu]
- `--setting`: Cross-subject (cs) or cross-view (cv) [default: cv]

### Evaluation Configuration
- `--eval-model`: Evaluation model (sgn, mixformer, both) [default: sgn]
- `--no-eval`: Skip evaluation step
- `--use-validation`: Use validation set from comprehensive data based on camera
- `--slurm`: Submit evaluation job to SLURM
- `--email`: Email for SLURM notifications

### Visualization Configuration
- `--visualize`: Enable visualization
- `--viz-types`: Types of visualizations (skeleton_animations, comparison_visualization, motion_visualization, attention_visualization, mlm_pretraining, all)

### Output Configuration
- `--output-dir`: Output directory for results

### Utility Options
- `--interactive`: Run in interactive mode
- `--list-models`: List available model files
- `--validate-path`: Validate a model path
- `--search-dir`: Directory to search for models

## Usage Examples

### 1. List Available Models

```bash
python scripts/standalone_eval.py --list-models
```

This will recursively search for model files and display them with inferred types.

### 2. Validate Model Path

```bash
python scripts/standalone_eval.py --validate-path /path/to/model.pth
```

### 3. Interactive Model Search

```bash
python scripts/standalone_eval.py --interactive --search-dir output/
```

### 4. Comprehensive Evaluation

```bash
python scripts/standalone_eval.py \
    --model-path trained_models/transformer_ntu_cv.pth \
    --model-type transformer \
    --dataset ntu \
    --setting cv \
    --eval-model sgn \
    --visualize \
    --viz-types skeleton_animations comparison_visualization \
    --output-dir results/comprehensive_eval
```

### 5. Visualization Only

```bash
python scripts/standalone_eval.py \
    --model-path model.pth \
    --model-type transformer \
    --dataset ntu \
    --setting cv \
    --no-eval \
    --visualize \
    --viz-types all
```

### 6. Evaluate All Available Models

```bash
python scripts/standalone_eval.py \
    --model-type all \
    --dataset ntu \
    --setting cv \
    --eval-model sgn \
    --output-dir results/all_models_evaluation
```

### 7. SLURM Evaluation with Validation Set

```bash
python scripts/standalone_eval.py \
    --model-path model_all.pth \
    --model-type transformer \
    --dataset ntu \
    --setting cv \
    --slurm \
    --use-validation \
    --email your.email@example.com \
    --output-dir results/slurm_validation_eval
```

### 8. Auto-find Default Model

```bash
python scripts/standalone_eval.py \
    --model-path auto \
    --model-type transformer \
    --dataset ntu \
    --setting cv \
    --eval-model sgn
```

## Model Type Inference

The script automatically infers model types based on file paths:

- **DMR**: Files containing 'dmr' in the path
- **PMR**: Files containing 'pmr' in the path  
- **Transformer**: Files containing 'transformer', 'model.pth', or in output/trained_models directories
- **Unknown**: Will prompt user in interactive mode

## Output Structure

Results are saved in the specified output directory:

```
output_dir/
├── evaluation_report.md          # Comprehensive report
├── evaluation_results.json       # Raw evaluation metrics (if available)
└── visualizations/               # Generated visualizations
    ├── skeleton_animations/
    ├── comparisons/
    └── ...
```

## Interactive Mode Features

### Model Selection
1. **Direct Path Entry**: Enter model path manually
2. **Current Directory Search**: Search for models in current directory
3. **Custom Directory Search**: Search in specified directory

### Path Validation
- Checks file existence
- Validates file extensions (.pth, .pt, .tar, .pkl)
- Infers model type automatically

### Configuration Wizard
- Dataset selection with defaults
- Setting selection (cs/cv)
- Evaluation model choice (SGN/MixFormer)
- Output directory configuration
- Evaluation/visualization toggles

## Integration with Existing Pipeline

The standalone script uses the same evaluation functions as the main pipeline:
- `eval_model.py` for core evaluation
- `evaluation_suite/run_visualization.py` for visualizations
- Same model loading and data processing logic

## Troubleshooting

### Common Issues

1. **Model Path Not Found**
   - Use `--list-models` to find available models
   - Use `--validate-path` to check specific paths

2. **Model Type Inference Failed**
   - Specify `--model-type` explicitly
   - Use interactive mode for guided selection

3. **Evaluation Errors**
   - Check that required model weights exist
   - Verify dataset and setting compatibility
   - Check logs in `logs/standalone_eval.log`

4. **Visualization Errors**
   - Ensure visualization dependencies are installed
   - Check output directory permissions
   - Try with fewer visualization types

### Log Files

- Main log: `logs/standalone_eval.log`
- Evaluation logs: Standard eval_model.py logging
- Visualization logs: Standard evaluation_suite logging

## Dependencies

The script uses existing project dependencies:
- Core evaluation: `eval_model.py`
- Visualization: `evaluation_suite/`
- Standard Python libraries: `os`, `sys`, `argparse`, `glob`, `json`

No additional dependencies required beyond the main project requirements.
