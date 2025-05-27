# Executive Summary: MLM Pretraining Evaluation Strategy

**Date:** Current  
**Meeting:** 30 minutes  
**Status:** Implementation Plan Ready

## Problem Identified

The current MLM pretraining evaluation has fundamental issues:
- **Architecture Mismatch**: MLM trained for coordinate reconstruction, evaluated for classification
- **Data Format Issues**: Output format incompatible with recognition models  
- **Conceptual Flaw**: Reconstruction quality ≠ Classification performance
- **Poor Results**: AR accuracy ~2%, RI accuracy ~4% across all masking ratios

## Root Cause

MLM models optimize for spatial coordinate reconstruction, not semantic feature preservation needed for action recognition and person identification.

## New Evaluation Strategy

### Core Approach
**Use encoder features for classification instead of reconstructed coordinates**

### Implementation Plan

#### Phase 1: Feature-Based Classification (Next 2 hours)
1. **Extract encoder embeddings** from pretrained MLM models
2. **Train lightweight classifiers** on embedding space
3. **Separate models** for Action Recognition (AR) and Re-Identification (RI)
4. **Cross-view split** for train/test (camera-based)

#### Phase 2: Comprehensive Evaluation Suite
1. **9 MLM models** (3 temporal × 3 spatial masking ratios)
2. **18 classifiers** (9 models × 2 tasks each)
3. **Physical plausibility metrics** retained from previous evaluation
4. **Publication-ready** reports and visualizations

#### Phase 3: HPC Automation
1. **Training script** for each MLM model's classifiers
2. **Queue script** for all 9 jobs (1 GPU each)
3. **Report generation** script for comprehensive analysis
4. **Integration** with evaluation_suite/

### Technical Details

#### Classifier Architecture
```python
# Simple MLP on encoder features
class MLMClassifier(nn.Module):
    def __init__(self, input_dim=320, num_classes=60):
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
```

#### Evaluation Metrics
- **Classification**: Accuracy, F1-score, Confusion matrices
- **Physical Plausibility**: BLC, JAL, TS, VC, FCC (retained)
- **Reconstruction**: MSE between input/output coordinates
- **Comparative Analysis**: Across all 9 masking combinations

#### Expected Deliverables
1. **Trained classifiers** for all 9 MLM models
2. **Comprehensive evaluation** results
3. **Publication-quality** graphs and tables
4. **HPC automation** scripts
5. **Integration** with evaluation suite

### Timeline
- **Phase 1**: 2 hours (classifier training)
- **Phase 2**: 4 hours (comprehensive evaluation)
- **Phase 3**: 2 hours (HPC scripts + integration)
- **Total**: ~8 hours for complete implementation

### Benefits
1. **Proper evaluation** of MLM embedding quality
2. **Meaningful metrics** for pretraining effectiveness
3. **Publication-ready** results
4. **Automated pipeline** for future experiments
5. **Maintains** physical plausibility analysis

## Next Steps
1. Implement feature extraction from MLM encoders
2. Create classifier training pipeline
3. Develop HPC automation scripts
4. Generate comprehensive evaluation reports

**Ready to proceed with implementation.**
