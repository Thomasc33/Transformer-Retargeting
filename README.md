# Transformer-Retargeting

This repository contains the official implementation of our skeleton motion retargeting framework using transformer models. Our approach leverages spatial-temporal features from skeleton sequences to transfer human motion patterns between different actors while preserving action characteristics.

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


