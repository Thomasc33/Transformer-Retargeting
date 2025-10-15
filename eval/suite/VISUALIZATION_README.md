# Visualization Integration for Transformer Retargeting

This document describes the enhanced visualization capabilities added to the evaluation suite and pipeline integration.

## Overview

The visualization system provides high-quality skeleton animations with proper Kinect v2 view angles, similar to the MLM visualization rendering but enhanced for transformer retargeting evaluation.

## Key Features

### 1. Enhanced Skeleton Animations
- **Proper Kinect v2 View Angles**: Optimized camera positioning (elevation=20°, azimuth=45°)
- **High-Quality Rendering**: Clean 3D visualization with proper bone connections
- **Multiple Animation Types**: Original, retargeted, sensitivity analysis, masking, noise
- **Consistent Scaling**: Global bounds calculation for consistent frame-to-frame scaling

### 2. Visualization Types

#### Skeleton Animations
- `original_skeleton`: Original motion sequences
- `retargeted_skeleton`: Anonymized/retargeted sequences
- `side_by_side_comparison`: Original vs retargeted side-by-side
- `overlay_comparison`: Overlaid skeletons with transparency
- `sensitivity_analysis`: Joint importance visualization
- `smart_masking`: Masked joint visualization
- `smart_noise`: Noise-based anonymization
- `group_noise`: Group-based noise application
- `naive_noise`: Random noise baseline

#### Comparison Visualizations
- `original_vs_all_methods`: Compare all anonymization methods
- `method_effectiveness_grid`: Grid layout comparison
- `privacy_utility_tradeoff`: Privacy vs utility analysis
- `temporal_consistency_analysis`: Motion smoothness evaluation
- `joint_importance_heatmaps`: Importance score visualization

### 3. Technical Implementation

#### Core Components
- `VisualizationExperiments`: Configuration and core visualization functions
- `VisualizationEvaluator`: Experiment execution and model integration
- `run_visualization.py`: Command-line interface for pipeline integration

#### Key Functions
- `create_skeleton_figure()`: Kinect v2 optimized figure creation
- `setup_3d_axis_kinect()`: Proper 3D axis configuration
- `draw_skeleton_3d()`: Enhanced skeleton rendering with importance scores
- `create_skeleton_animation()`: Main animation creation function

## Pipeline Integration

### 1. Updated Pipeline Steps
The pipeline now includes a `visualize` step:
```bash
python scripts/pipeline.py --steps sample,train,evaluate,visualize --dataset ntu --setting cv
```

### 2. Interactive Configuration
When running interactively, users can select from:
- Skeleton Animations
- Motion Visualizations
- Attention Visualization
- Comparison Visualizations
- Sensitivity Analysis
- Anonymization Showcase

### 3. Command Line Usage
```bash
# Run specific visualizations
python evaluation_suite/run_visualization.py --visualizations skeleton_animations --dataset ntu --setting cv

# Run all visualizations
python evaluation_suite/run_visualization.py --visualizations all --dataset ntu --setting cv

# Run multiple visualizations
python evaluation_suite/run_visualization.py --visualizations skeleton_animations,comparison_visualizations --dataset ntu --setting cv
```

## Configuration

### Visualization Settings
```python
'view_settings': {
    'camera_angle': {
        'elevation': 20,
        'azimuth': 45
    },
    'kinect_v2_view': True,
    'fixed_bounds': True,
    'aspect_ratio': 'equal'
}
```

### Quality Settings
```python
'quality_settings': {
    'fps': 10,
    'duration_per_frame': 0.15,
    'resolution': (800, 600),
    'joint_size': 50,
    'bone_width': 2
}
```

## Output Structure

Visualizations are saved to `results/visualizations/` with the following structure:
```
results/visualizations/
├── skeleton_animations/
│   ├── transformer/
│   │   ├── original_animation.gif
│   │   ├── retargeted_animation.gif
│   │   └── sensitivity_animation.gif
│   ├── dmr/
│   └── pmr/
├── comparison_visualizations/
└── motion_visualizations/
```

## Kinect v2 View Angle Reference

The visualization system uses the optimal Kinect v2 view angles from your reference code:
- **Elevation**: 20° (slight upward angle)
- **Azimuth**: 45° (diagonal view)
- **Clean axes**: No grid, minimal ticks
- **Transparent panes**: Clean background
- **Equal aspect ratio**: Proper skeleton proportions

## Testing

Run the test suite to verify functionality:
```bash
python evaluation_suite/test_visualization.py
```

Tests include:
- Skeleton figure creation
- Data reshaping
- Skeleton drawing
- Animation creation
- Evaluator initialization

## Integration with Existing MLM Visualizations

The new visualization system:
- **Extends** the existing MLM visualizer functionality
- **Maintains compatibility** with existing data formats
- **Adds enhanced features** like importance scoring and masking
- **Uses consistent** color schemes and styling
- **Provides better** Kinect v2 view angles

## Usage Examples

### Basic Skeleton Animation
```python
from evaluation_suite.experiments.visualization import VisualizationExperiments

# Create animation
animation_path = VisualizationExperiments.create_skeleton_animation(
    samples, 
    output_dir='my_visualizations',
    figure_type='original',
    max_frames=50
)
```

### Full Evaluation Suite
```python
from evaluation_suite.core.visualization_evaluator import VisualizationEvaluator

evaluator = VisualizationEvaluator()
results = evaluator.run_all_visualization_experiments()
```

## Future Enhancements

Planned improvements:
1. **Interactive 3D visualizations** using Plotly
2. **Video export** in MP4 format
3. **Batch processing** for large datasets
4. **Custom view angles** configuration
5. **Real-time visualization** during training

## Dependencies

Required packages:
- `matplotlib` (3D plotting)
- `imageio` (GIF creation)
- `numpy` (data processing)
- `torch` (model integration)
- `tqdm` (progress bars)

## Notes

- Visualizations are optimized for HPC environments (no interactive displays)
- All outputs are saved as files (GIFs, images)
- Memory usage is optimized for large datasets
- Compatible with both CPU and GPU processing
