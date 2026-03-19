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
This repository provides the official implementation of **BEV-SLD**, including all scripts required to reproduce our experiments on the **Wild-Places** and **MCD dataset**.

For a quick test with a pretrained model, see the [`demo` branch](https://github.com/davidskdds/BEV-SLD/tree/demo). Preprocessed BEV images are included, so no dataset download is required for the demo.

# Setup & Requirements
We provide a bash script for setting up a virtual environment.
Before creating the environment, please adjust the PyTorch version in `requirements.txt` to match your local CUDA version.  
Then, to create a virtual environment named `bev_sld_env` locally, run:  
`bash create_env.sh`


# Datasets

## MCD Dataset

Download the following reference and test sequences from the official MCD dataset website:
[https://mcdviral.github.io/Download.html](https://mcdviral.github.io/Download.html)

For our experiments, we used the Ouster ROS bag files.

Reference/training sequence:

* `ntu_day_01_os1_128.bag`

Test sequences:

* `ntu_day_02_os1_128.bag`
* `ntu_day_10_os1_128.bag`
* `ntu_night_04_os1_128.bag`
* `ntu_night_08_os1_128.bag`
* `ntu_night_13_os1_128.bag`

The MCD dataset provides ground-truth poses in the vehicle frame. We transform them into the Ouster LiDAR frame using the calibration parameters released by the dataset authors. These transformed poses, which are used for training and evaluation, are stored under `gt_poses/mcd/`.

We provide two example configuration files: `config/mcd_ntu_day_01_ref.yaml` and `config/mcd_ntu_day_10.yaml`. Additional sequence configuration files can be created in the same way.

## Wild-Places Dataset

Download the following reference and test sequences from the official Wild-Places dataset website:
[https://data.csiro.au/collection/csiro:56372](https://data.csiro.au/collection/csiro:56372)

More information is available on the project website:
[https://csiro-robotics.github.io/Wild-Places/](https://csiro-robotics.github.io/Wild-Places/)

Reference/training sequence:

* `V-03`

Test sequences:

* `V-01`
* `V-02`
* `V-04`

For the Wild-Places dataset, the point clouds are already provided as `.pcd` files, so no ROS bag extraction is required. We provide two example configuration files: `config/wild_places_v03_ref.yaml` and `config/wild_places_v01.yaml`. Additional sequence configuration files can be created in the same way.


# Configuration
All configuration files are provided as `.yaml` files under `config/`.

- Each .yaml defines paths, hyperparameters, and dataset settings.
- Command-line arguments (via `argparse`) override `.yaml` parameters.
- Before preprocessing, ensure the bag_path parameter points to the correct rosbag file.

For e.g. the reference sequence settings in `config/mcd_ntu_day_01_ref.yaml` change the parameter `bag_path` to your rosbag path.
For the configuration of Wild-Places dataset, no Rosbags are provided but .pcd files. Therefore, in `config/wild_places_v03_ref.yaml` the parameter `pc_dir` must be adjusted.

# Preprocessing
Check if you actived the virtual environment. If not, run:  
`source bev_sld_env/bin/activate`

Convert ROS bag data to `.pcd` point clouds (this step can omitted for the Wild-Places dataset):  
`python extract_pcs_rosbag.py --config config/mcd_ntu_day_01_ref.yaml`

Create BEV images and global coordinate maps (saved as `.tif` files):  
`python create_bev_images_and_coord_maps.py --config config/mcd_ntu_day_01_ref.yaml`

These preprocessing steps need to be performed only once per sequence.

# Training
To train the BEV-SLD model on the reference sequence, run:  
`python train.py --config config/mcd_ntu_day_01_ref.yaml`

# Localization
To use a trained network for localization, run:  
`python localization.py --config config/mcd_ntu_day_01_ref.yaml`

# Evaluation
To evaluate the success rate (SR) and median errors, run:  
`python eval_poses.py --config config/mcd_ntu_day_01_ref.yaml`

To evaluate the model on a test sequence, follow the same steps as above, excluding the training stage. Run the scripts with:  
`--config config/mcd_ntu_day_10.yaml`