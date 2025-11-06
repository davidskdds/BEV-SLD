import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,DataLoader,random_split
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
import open3d as o3d
from network import bev_sld_net
from random import randrange
import random
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from matplotlib import pyplot as plt
import torch.nn as nn
from utils import get_config,read_poses,get_lr,save_config_as_yaml
import tifffile
from augment import transform_data

# tensorboard --logdir=runs/ --host localhost --port 8088 --reload_multifile True
# browser: http://localhost:8088

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import math

plot_landmarks = False
Save_Landmarks = True


# generate initial landmarks from training dataset
def get_initial_lms_dataset(train_loader,device_in,cfg):
    
    valid_landmarks = torch.zeros(1, 2, device=device_in)
    
    # deactivate transform for initial landmarks generation
    train_loader.dataset.dataset.do_transform = False
    
    with torch.no_grad():
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device_in), labels.to(device_in)

            valid_landmarks = get_initial_lms_batch(labels,inputs,cfg.n_div,cfg.n_padding,valid_landmarks)
    
    # activate transform again
    train_loader.dataset.dataset.do_transform = True        
    
    initial_landmarks_np = valid_landmarks.cpu().detach().numpy() 
    
    
    # add random offset to each point
    initial_landmarks_np += (2.0*np.random.rand(initial_landmarks_np.shape[0],2) -1.0)* 0.5*cfg.grid_res *cfg.n_xy / cfg.n_div
    
    # downsampling
    # Step 2: Add z=0 to each point -> becomes (n x 3)
    points_3d = np.hstack((initial_landmarks_np, np.zeros((initial_landmarks_np.shape[0], 1))))

    # Step 3: Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Step 4: Voxel downsampling
    voxel_size = 1.0/np.sqrt(cfg.lm_density)*cfg.n_xy / cfg.n_div * cfg.grid_res  # adjust as needed
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Step 5: Access downsampled points
    downsampled_initial_landmarks = np.asarray(pcd_down.points)[:,0:2]
    
    return downsampled_initial_landmarks

# generate initial landmarks from a batch
def get_initial_lms_batch(target,density_imgs,n,padding,valid_landmarks):

    nrc = int(density_imgs.shape[3] / n)
    error_sum = 0.0
    weight_sum = 0
    loss_sum = 0.0
    n_patches = 0

    for row in range(0+padding,n-padding):
        for col in range(0+padding,n-padding):
            start_row = row * nrc
            start_col = col * nrc
            
            curr_target = target[:,:,start_row:start_row+nrc,start_col:start_col+nrc]
            curr_density_images = density_imgs[:,:,start_row:start_row+nrc,start_col:start_col+nrc]

            density_per_batch = torch.sum(curr_density_images,dim=(1,2,3))

            valid_density = torch.ge(density_per_batch,1e-5)

            # flatten
            curr_target = torch.flatten(curr_target,start_dim=2, end_dim=3)

            mean_targetX = torch.mean(curr_target[:,0:1,:],dim=2)
            mean_targetY = torch.mean(curr_target[:,1:2,:],dim=2)
            
            pred_landmarks_xy = torch.cat((mean_targetX, mean_targetY), dim=1)

            valid_landmarks = torch.cat((valid_landmarks, pred_landmarks_xy[valid_density,:]), dim=0) 


    return valid_landmarks


def landmark_location_and_corresp_loss(heat_map, corresp, coords, global_coordinate_map,density_imgs,cfg):
    """
    Compute the combined landmark location/distance and correspondence loss.

    This function estimates landmark positions from predicted heatmaps, 
    compares them to ground-truth coordinates, and enforces correspondence 
    consistency between predicted and true landmarks. It combines a 
    distance-based localization loss and a cross-entropy correspondence loss.

    # Arguments
    heat_map (torch.Tensor): 
        Predicted landmark heatmap of shape (B, 1, H, W), 
        where B is batch size.
    corresp (torch.Tensor): 
        Predicted landmark correspondence logits of shape (B, L, H/32, W/32),
        where L is the number of landmark classes.
    coords (torch.Tensor): 
        Trainable landmark coordinates of shape (L, 2), 
        containing (x, y) positions in global frame.
    global_coordinate_map (torch.Tensor): 
        Ground-truth coordinate grid of shape (B, 2, H, W), 
        containing the global x and y coordinates for each pixel.
    density_imgs (torch.Tensor): 
        Continuous density maps of shape (B, 1, H, W), 
        marking valid landmark regions.
    cfg (object): 
        Configuration object containing required hyperparameters:
            - n_div (int): division factor for patch extraction.
            - n_padding (int): number of patches to pad around borders.
            - grid_res (float): spatial grid resolution.
            - loss_gamma (float): weighting factor for distance loss.
            - loss_alpha (float): weight for landmark distance loss.
            - loss_beta (float): weight for correspondence loss.

    # Returns
    combined_loss (torch.Tensor): 
        Weighted sum of the landmark distance and correspondence loss.
    lm_dist_error (torch.Tensor): 
        Mean landmark distance error term (log-scaled and weighted).
    ce_error (torch.Tensor): 
        Cross-entropy loss for landmark correspondences.

    # Notes
    - The heatmap is divided into patches via `torch.nn.functional.unfold`.
    - The predicted landmark is computed as the heatmap-weighted mean of (x, y).
    - Landmark distance errors beyond a patch-dependent threshold are excluded.
    - The final combined loss = α * lm_dist_error + β * ce_error.
    """
    
    # n = 16
    nrc = int(heat_map.shape[3] / cfg.n_div)
    weight_sum = 0
    loss_sum = 0.0
    
    b = heat_map.shape[3] // cfg.n_div
    b_corresp = corresp.shape[3] // cfg.n_div
    
    # new shapes: (B, b*b, n*n)
    from_padding = cfg.n_padding * b
    to_padding = cfg.n_div * b - cfg.n_padding * b
    corresp_flat = corresp[:,:,cfg.n_padding:-cfg.n_padding,cfg.n_padding:-cfg.n_padding].flatten(2)
    corresp_flat = corresp_flat.permute(0, 2, 1)        # now [B, P, C]
    corresp_flat = corresp_flat.reshape(-1, corresp_flat.size(2))  # collapses B and P → [B*P, C]

    heatmap_patches = F.unfold(heat_map[:,:,from_padding:to_padding,from_padding:to_padding], kernel_size=b, stride=b)
    density_patches = F.unfold(density_imgs[:,:,from_padding:to_padding,from_padding:to_padding], kernel_size=b, stride=b)
    x_patches = F.unfold(global_coordinate_map[:,0:1,from_padding:to_padding,from_padding:to_padding], kernel_size=b, stride=b)
    y_patches = F.unfold(global_coordinate_map[:,1:2,from_padding:to_padding,from_padding:to_padding], kernel_size=b, stride=b)
    
    # normalize
    heatmap_patches = nn.functional.softmax(heatmap_patches,dim=1)
    
    # estimated landmarks
    x_pred = torch.flatten(torch.sum(heatmap_patches * x_patches, dim=1))
    y_pred = torch.flatten(torch.sum(heatmap_patches * y_patches, dim=1))
    
    valid_density = torch.ge(torch.flatten(torch.sum(density_patches, dim=1)),1e-5)
    
    sq_err_x_lm = torch.square(torch.sub(x_pred[:,None],coords[:,0][None,:]))
    sq_err_y_lm = torch.square(torch.sub(y_pred[:,None],coords[:,1][None,:]))
    

    # sqrt for l1 error
    sq_error = torch.add(sq_err_x_lm,sq_err_y_lm)
    abs_error_lm = torch.sqrt( sq_error.clamp(min=1e-6))

    # find minimum error to a landmark for each patch
    min_err,ids = torch.min(abs_error_lm[valid_density,:],dim=1)
    
    # valid correspondences for loss
    corresp_flat_valid = corresp_flat[valid_density,:]
    
    # corresp error
    ce_error = nn.functional.cross_entropy(corresp_flat_valid,ids)
    
    # log scale
    min_err = torch.log(cfg.loss_gamma*min_err + 1.0)

    # normalized
    lm_dist_error = torch.sum(min_err) / ( 1 + torch.sum(valid_density) )

    combined_loss = cfg.loss_alpha*lm_dist_error + cfg.loss_beta*ce_error

    return combined_loss, lm_dist_error, ce_error


class LandmarkDetDataset(Dataset):
    def __init__(self, input_dir, label_dirs, transform=None, dtype=np.float32):
        """
        Args:
            input_dir (str): path to folder with single-channel .tif inputs
            label_dirs (list of str): two folders, each with one .tif per input,
                                      e.g. ["/.../label_x", "/.../label_y"]
            transform (callable, optional): transforms to apply to both image and label
            dtype (np.dtype): dtype for the numpy arrays (defaults to float32)
        """
        self.input_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('.tif', '.tiff'))
        ])
        # assume same filenames in each label dir
        self.label_dirs = label_dirs
        self.transform = transform
        self.dtype = dtype
        self.do_transform = True

        # build parallel lists of label paths
        self.label_paths = []
        for ld in label_dirs:
            paths = sorted([
                os.path.join(ld, f)
                for f in os.listdir(ld)
                if f.lower().endswith(('.tif', '.tiff'))
            ])
            assert len(paths) == len(self.input_paths), \
                f"Expected {len(self.input_paths)} labels in {ld}, found {len(paths)}"
            self.label_paths.append(paths)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        # load input image
        inp = tifffile.imread(self.input_paths[idx]).astype(self.dtype)
        # load and stack both labels into shape (2, H, W)
        labs = []
        for lp_list in self.label_paths:
            l = tifffile.imread(lp_list[idx]).astype(self.dtype)
            labs.append(l)
        label = np.stack(labs, axis=0)

        # to torch tensors: input as (1, H, W), label as (2, H, W)
        inp_t = torch.from_numpy(inp).unsqueeze(0)
        lbl_t = torch.from_numpy(label)

        # optional joint transforms (e.g. random crop, flip)
        if self.do_transform:
            if self.transform is not None:
                inp_t, lbl_t = self.transform(inp_t, lbl_t)

        return inp_t, lbl_t


def main():
    writer = SummaryWriter()
    plot_landmarks = False
    Save_Landmarks = True

    # read config
    cfg = get_config()

    # save
    save_config_as_yaml(cfg, 'config.yaml')

    # read poses
    curr_poses = read_poses(cfg.pose_file_dir)

    # create network folder if it doesnt exist
    network_dir = os.path.dirname(cfg.network_path)
    os.makedirs(network_dir, exist_ok=True)

    # results dir
    result_dir = cfg.dataset_dir + '/results'+cfg.result_dir + '/'
    os.makedirs(result_dir, exist_ok=True)

    # device
    device = torch.device('cuda:'+str(cfg.cuda_id) if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # create dataset
    dataset = LandmarkDetDataset(cfg.dataset_dir + '/bev_images', [cfg.dataset_dir + '/label_x', cfg.dataset_dir + '/label_y'],transform=transform_data)

    print('Num samples:', len(dataset))

    # split in train and test
    total_size = len(dataset)
    test_size  = int(cfg.test_frac * total_size)
    train_size = total_size - test_size

    # 3. random split (uses a default RNG; pass `generator=torch.Generator().manual_seed(42)` for reproducibility)
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # 4. wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=16, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # deactivate transforms for test loader
    test_loader.dataset.dataset.do_transform = False  

    # create init landmarks
    print('Create initial landmarks . . .')
    initial_landmarks = get_initial_lms_dataset(train_loader,device,cfg)

    if plot_landmarks:
        plt.ion()
        plt.show()
        plt.scatter(curr_poses[:,1], curr_poses[:,2])
        plt.scatter(initial_landmarks[:,0], initial_landmarks[:,1])
        plt.draw()
        plt.pause(0.001)

    # init network
    print('Num landmarks: ',initial_landmarks.shape[0])
    initial_landmarks_cpy = initial_landmarks.copy()

    # initialize model with initial landmarks
    print('Initialize model . . .')
    model = bev_sld_net(torch.from_numpy(initial_landmarks_cpy)).to(device)
        
    # print number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num model params: {pytorch_total_params}")

    loss_idx_value = 0
    bestEpoch = 0

    # init
    loss_fn = landmark_location_and_corresp_loss
    optimizer = optim.SGD(model.parameters(), lr=cfg.start_lr,momentum=0.9)

    target_factor =  cfg.final_lr / cfg.start_lr
    decr_per_epoch = target_factor ** (1.0/cfg.num_epochs)

    scheduler = StepLR(optimizer, step_size=1, gamma=decr_per_epoch)

    np.save(result_dir+'initial_landmarks.npy', initial_landmarks)

    if Save_Landmarks is True:
        save_lm_dir = result_dir + 'landmarks_per_ep/'
        os.makedirs(save_lm_dir, exist_ok=True)

    lm_idx = 0

    for epoch in range(cfg.num_epochs):

        # train one epoch
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            heat_map, corresp, coords = model(inputs)

            loss, lm_dist_error, ce_error = loss_fn(heat_map,corresp,coords,labels,inputs,cfg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if Save_Landmarks is True:
                curr_coords = coords.cpu().detach().numpy()[:,:]
                x, y = curr_coords.T
                save_lm_dir = result_dir + 'landmarks_per_ep/'
                curr_lm_file = save_lm_dir + 'landmarks_epoch_' + str(lm_idx) + '.npy'
                
                np.save(curr_lm_file, curr_coords)
                lm_idx = lm_idx + 1


        # test dataset
        torch.cuda.empty_cache()

        # check validation data
        loss_sum = 0.0
        n_test = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:

                inputs, labels = inputs.to(device), labels.to(device)

                heat_map, corresp, coords = model(inputs)

                loss, lm_dist_error, ce_error = loss_fn(heat_map, corresp, coords,labels,inputs,cfg)

                loss_sum += loss
                n_test += 1.0
            
            curr_coords = coords.cpu().detach().numpy()[:,:]
            x, y = curr_coords.T

            res = epoch%3
            if plot_landmarks:
                if res == 0:
                    plt.cla() 
                plt.scatter(x, y)
                plt.draw()
                plt.pause(0.001)

        
        val_loss = loss_sum / n_test
        writer.add_scalar("Loss/combined_loss", val_loss, epoch)
        writer.add_scalar("Loss/lm_distance_loss", lm_dist_error, epoch)
        writer.add_scalar("Loss/corresp_ce_loss", ce_error, epoch)
        writer.add_scalar("Loss/lr", get_lr(optimizer), epoch)
        print('landmarks max / min x: ', np.max(curr_coords[:,0]), ' / ',np.min(curr_coords[:,0]))
        print('landmarks max / min y: ', np.max(curr_coords[:,1]), ' / ',np.min(curr_coords[:,1]))
        writer.flush()   

        # update scheduler
        scheduler.step()    

        torch.cuda.empty_cache()

        print("Epoch:", epoch)
        print("Best epoch:", bestEpoch)
        print("Valid Loss:", val_loss)

        if epoch == 0 or val_loss < minLoss:
            minLoss = val_loss
            torch.save(model, cfg.network_path)
            torch.save(model, result_dir+'det.pth')
            
            best_model = model
            print("Saved new model")
            bestEpoch = epoch


if __name__ == "__main__":
    main()
