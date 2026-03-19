<div align="center">
<h1>BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images</h1>

<a href="https://arxiv.org/abs/2603.17159"><img src="https://img.shields.io/badge/arXiv-2603.17159-b31b1b" alt="arXiv"></a>

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


This repository provides a **BEV-SLD** demo that showcases localization on samples from the **MCD dataset** (https://mcdviral.github.io/).

# Setup & Requirements
We provide a bash script to set up a virtual environment.  
Before creating the environment, please adjust the PyTorch version in `requirements.txt` to match your local CUDA version.  
Then, to create a virtual environment named `bev_sld_env` locally, run:  
`bash create_env.sh`

# Run Demo
In this demo, we provide a pretrained network trained on the sequence ntu_day_01 of the MCD dataset.
The pretrained model is located at:  
`datasets/mcd/ntu_day_01_os1/models/bev_sld_model.pth`  

To evaluate this network, we also provide 60 BEV images from the sequence ntu_day_10, located at:   
`datasets/mcd/ntu_day_10_os1/bev_images`  

First, activate the virtual environment created in the previous step:   
`source bev_sld_env/bin/activate`  

Then, run the localization demo:  
`python localization.py --config config/mcd_ntu_day_10.yaml`  

A plot showing the landmarks and the estimated poses will appear.  
**The full code, including training scripts, will be released upon paper acceptance.**
