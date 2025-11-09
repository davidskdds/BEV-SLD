# BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird’s-Eye View Images

<p align="center"> <img src="./fig/localization.gif" alt="Global localization on the MCD dataset" width="900"/> </p> <p align="center"><em>Fig. 1 — Global localization on the MCD dataset.</em></p>

This repository provides the official implementation of **BEV-SLD**, including all scripts required to reproduce our experiments on the **MCD dataset**.

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
`python localization.py --config config/mcd_ntu_day_10_map.yaml`  

A plot showing the landmarks and the estimated poses will appear.  
**The full code will be released upon paper acceptance.**
