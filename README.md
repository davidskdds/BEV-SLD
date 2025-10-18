# BEV-SLD
We provide code to be able to reproduce our experiments on the MCD-dataset.


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
We provide `.yaml` files that can be used for training/test configurations.
In addition, parameters can also be set using argparse. argparse parameters overwrite `.yaml` parameter settings.
Before preprocessing can start, the rosbag file paths have to be added.
Therefore add the path where the downloaded rosbag is to your `.yaml` file.

E.g. in `config/mcd_ntu_day_01_map.yaml` change the parameter bag_path to your rosbag path.

# Preprocessing
First, we extract .pcd point clouds from the rosbags:  
`python extract_pcs_rosbag.py --config config/mcd_ntu_day_01_map.yaml`

Then, bev images and global coordinate maps are generated and saved as .tif images:  
`python create_bev_images_and_coord_maps.py --config config/mcd_ntu_day_01_map.yaml`

The preprocessing steps have to be performed only once for each dataset sequence.

# Training
To start the training on the mapping sequence, simply enter:  
`python train.py --config config/mcd_ntu_day_01_map.yaml`