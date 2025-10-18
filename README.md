# BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird’s-Eye View Images

<figure>
<p align='left'>
<img src="./fig/localization.gif" alt="drawing" width="900"/>
</p>
<figcaption>Fig. 1 - Global localization on the MCD-dataset.</figcaption>
</figure>


This repository provides the official implementation of **BEV-SLD**, including all scripts required to reproduce our experiments on the **MCD dataset**.

# Requirements
1. Create a conda environment
2. Install the pytorch / torchvision versions matching your cuda version
3. Install packages from requirements.txt


# Dataset
Download the following mapping and test sequences from the official MCD-dataset website: https://mcdviral.github.io/Download.html  
For our experiments, we used the Ouster rosbags.

Map/Training sequence:
- ntu_day_01_os1_128.bag

Test sequences:
- ntu_day_02_os1_128.bag
- ntu_day_10_os1_128.bag
- ntu_night_04_os1_128.bag
- ntu_night_08_os1_128.bag
- ntu_night_13_os1_128.bag

As all ground truth poses are defined in the vehicle frame, we used the calibration parameters provided by the authors of the dataset to transform the ground truth poses to the Ouster LiDAR frame. This poses in the Ouster frame we used for training/evaluation.
The poses are provided in `gt_poses/mcd`.

# Configuration
All configuration files are provided as `.yaml` files under `config/`.

- Each .yaml defines paths, hyperparameters, and dataset settings.
- Command-line arguments (via `argparse`) override `.yaml` parameters.
- Before preprocessing, ensure the bag_path parameter points to the correct rosbag file.

For e.g. the mapping sequence settings in `config/mcd_ntu_day_01_map.yaml` change the parameter `bag_path` to your rosbag path.

# Preprocessing
Convert ROS bag data to `.pcd` point clouds:  
`python extract_pcs_rosbag.py --config config/mcd_ntu_day_01_map.yaml`

Create BEV images and global coordinate maps (saved as `.tif` files):  
`python create_bev_images_and_coord_maps.py --config config/mcd_ntu_day_01_map.yaml`

The preprocessing steps have to be performed only once for each dataset sequence.

# Training
To train the BEV-SLD model on the mapping sequence, run:  
`python train.py --config config/mcd_ntu_day_01_map.yaml`