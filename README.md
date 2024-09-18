# Skeleton-MixFormer [ACMMM 2023]
This repo is the official implementation for Skeleton-MixFormer: <u>Multivariate Topology Representation for Skeleton-based Action Recognition<u>


## Abstract
Vision Transformer, which performs well in various vision tasks, encounters a bottleneck in skeleton-based action recognition and falls short of advanced GCN-based methods. The root cause is that the current skeleton transformer depends on the self-attention mechanism of the complete channel of the global joint, ignoring the highly discriminative differential correlation within the channel, so it is challenging to learn the expression of the multivariate topology dynamically. To tackle this, we present Skeleton MixFormer, an innovative spatio-temporal architecture to effectively represent the physical correlations and temporal interactivity of the compact skeleton data.  Two essential components make up the proposed framework: 1) Spatial MixFormer. The channel-grouping and mix-attention are utilized to calculate the dynamic multivariate topological relationships. Compared with the full-channel self-attention method, Spatial MixFormer better highlights the channel groups' discriminative differences and the joint adjacency's interpretable learning. 2) Temporal MixFormer, which consists of Multiscale Convolution, Temporal Transformer and Sequential Holding Module. The multivariate temporal models ensure the richness of global difference expression and realize the discrimination of crucial intervals in the sequence, thereby enabling more effective learning of long and short-term dependencies in actions.
Our Skeleton MixFormer demonstrates state-of-the-art (SOTA) performance across seven different settings on four standard datasets, namely NTU-60, NTU-120, NW-UCLA, and UAV-Human.



# Dependencies

+ Python >= 3.6

+ PyTorch >= 1.1.0

+ PyYAML, tqdm, tensorboardX

# Data Preparation

### Download datasets

**There are 4 datasets to download:**
+ NTU RGB+D 60 Skeleton
+ NTU RGB+D 120 Skeleton
+ NW-UCLA
+ UAV-Human

**NTU RGB+D 60 and 120**

1. Request dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:  
    i. ```nturgbd_skeletons_s001_to_s017.zip``` (NTU RGB+D 60)  
    ii. ```nturgbd_skeletons_s018_to_s032.zip``` (NTU RGB+D 120)  
    iii. Extract above files to ```./data/nturgbd_raw```  


### NTU Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:
~~~
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
~~~

#### Generating Data

+ Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:
~~~
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
~~~

    
# Training & Testing
### Training
+ Change the config file depending on what you want.
~~~
    # Example: training SKMIXF on NTU RGB+D cross subject with GPU 0
    python main.py --config config/nturgbd-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/skmixf --device 0
    # Example: training provided baseline on NTU RGB+D cross subject
    python main.py --config config/nturgbd-cross-subject/default.yaml --model model.baseline.Model--work-dir work_dir/ntu/csub/baseline --device 0
~~~
+ To train model on NTU RGB+D 60/120 with bone or motion modalities, setting ```bone``` or ```vel``` arguments in the config file ```default.yaml``` or in the command line.
~~~
    # Example: training SKMIXF on NTU RGB+D 120 cross subject under bone modality
    python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --work-     dir work_dir/ntu120/csub/skmixf_bone --device 0
~~~
+ To train model on NW-UCLA with bone or motion modalities, you need to modify ```data_path``` in ```train_feeder_args``` and ```test_feeder_args``` to "bone" or "motion" or "bone motion", and run
~~~
    python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/skmixf_xxx --device 0
~~~
+ To train model on UAV-Human with bone or motion modalities, you need to modify ```data_path``` in ```train_feeder_args``` and ```test_feeder_args``` to "bone" or "motion" or "bone motion", and run
~~~
    python main.py --config config/uav/default.yaml --work-dir work_dir/uav/skmixf_xxx --device 0
~~~

### Testing

+ To test the trained models saved in <work_dir>, run the following command:  

~~~
    python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --         device 0
~~~

+ To ensemble the results of different modalities, run  

~~~
    # Example: ensemble four modalities of SkMIXF on NTU RGB+D cross subject
    python ensemble.py --dataset ntu/xsub  --joint-dir  work_dir/ntu/csub/skmixf --bone-dir  work_dir/ntu/csub/skmixf_bone --joint-motion-dir  work_dir/ntu120/csub/skmixf_motion  --bone-motion-dir work_dir/ntu/csub/skmixf_bone_motion  --joint-k2-dir work_dir/ntu120/csub/skmixf_joint_k2  --joint-motion-k2-dir  work_dir/ntu120/csub/skmixf_joint_motion_k2
~~~

### Pretrained model
+ Pretrained weights for NTU RGB+D 60 and 120 can be downloaded from the following link [[Google Drive]](https://drive.google.com/file/d/15Ahneq5_IgurficrYb3PiiLeEFyS8lBQ/view?usp=share_link)
    
## Acknowledgements
This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) and [Info-GCN](https://github.com/stnoah1/infogcn) The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!  


