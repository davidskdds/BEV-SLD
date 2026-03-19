<div align="center">
<h1>BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images</h1>

<a href="https://arxiv.org/abs/2603.17159"><img src="https://img.shields.io/badge/arXiv-2603.17159-b31b1b" alt="arXiv"></a>

[David Skuddis](https://www.ifp.uni-stuttgart.de/institut/team/Skuddis/), [Vincent Ress](https://www.ifp.uni-stuttgart.de/institut/team/Ress/), [Wei Zhang](https://willyzw.github.io/), [Vincent Ofosu Nyako](https://www.ifp.uni-stuttgart.de/institut/team/Ofosu-Nyako/), [Norbert Haala](https://www.ifp.uni-stuttgart.de/institut/team/Haala-00001/)

**[Institute for Photogrammetry and
Geoinformatics (ifp), University of Stuttgart](https://www.ifp.uni-stuttgart.de/en/)**

</div>

```bibtex
@inproceedings{skuddis2026bevsld,
  title={BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird’s-Eye View Images},
  author={David Skuddis and Vincent Ress and Wei Zhang and Vincent Ofosu Nyako and Norbert Haala},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
<p align="center"> <img src="./fig/localization.gif" alt="Global localization on the MCD dataset" width="900"/> </p> <p align="center"><em>Fig. 1 — Global localization on the MCD dataset.</em></p>

# Overview
This repository provides the official implementation of **BEV-SLD**, including all scripts required to reproduce our experiments on the **MCD dataset**.

For a quick test with a pretrained model, see the [`demo` branch](https://github.com/davidskdds/BEV-SLD/tree/demo). Preprocessed BEV images are included, so no dataset download is required for the demo.

# Setup & Requirements
We provide a bash script for setting up a virtual environment.
Before creating the environment, please adjust the PyTorch version in `requirements.txt` to match your local CUDA version.  
Then, to create a virtual environment named `bev_sld_env` locally, run:  
`bash create_env.sh`


# Dataset
Download the following mapping and test sequences from the official MCD dataset website:  
https://mcdviral.github.io/Download.html

For our experiments, we used the Ouster rosbags.

Map/Training sequence:
- ntu_day_01_os1_128.bag

Test sequences:
- ntu_day_02_os1_128.bag
- ntu_day_10_os1_128.bag
- ntu_night_04_os1_128.bag
- ntu_night_08_os1_128.bag
- ntu_night_13_os1_128.bag

The MCD dataset provides ground-truth poses in the vehicle frame.
We transform them into the Ouster LiDAR frame using the calibration parameters released by the dataset authors.
These transformed poses (used for training and evaluation) are stored under `gt_poses/mcd/`.

# Configuration
All configuration files are provided as `.yaml` files under `config/`.

- Each .yaml defines paths, hyperparameters, and dataset settings.
- Command-line arguments (via `argparse`) override `.yaml` parameters.
- Before preprocessing, ensure the bag_path parameter points to the correct rosbag file.

For e.g. the mapping sequence settings in `config/mcd_ntu_day_01_map.yaml` change the parameter `bag_path` to your rosbag path.

# Preprocessing
Check if you actived the virtual environment. If not, run:  
`source bev_sld_env/bin/activate`

Convert ROS bag data to `.pcd` point clouds:  
`python extract_pcs_rosbag.py --config config/mcd_ntu_day_01_map.yaml`

Create BEV images and global coordinate maps (saved as `.tif` files):  
`python create_bev_images_and_coord_maps.py --config config/mcd_ntu_day_01_map.yaml`

These preprocessing steps need to be performed only once per sequence.

# Training
To train the BEV-SLD model on the mapping sequence, run:  
`python train.py --config config/mcd_ntu_day_01_map.yaml`

# Localization
To use a trained network for localization, run:  
`python localization.py --config config/mcd_ntu_day_01_map.yaml`

# Evaluation
To evaluate the success rate (SR) and median errors, run:  
`python eval_poses.py --config config/mcd_ntu_day_01_map.yaml`

To evaluate the model on a test sequence, follow the same steps as above, excluding the training stage. Run the scripts with:  
`--config config/mcd_ntu_day_10.yaml`