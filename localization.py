import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from ransac_rigid_trafo import ransac_rigid_transform,ransac_3d
import torch.nn as nn
from random import randrange
import random
import torchvision.transforms.functional as TF
import sys
from scipy.spatial.transform import Rotation
from skimage.feature import peak_local_max

from utils import extract_timestamp
from timeit import default_timer as timer
from tqdm import tqdm
from utils import get_config,create_local_coord_map

def load_tiff_images_to_numpy(directory):
    image_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.tif')]
    image_paths.sort()
    images_list = []
    stamps = np.zeros((len(image_paths),1),dtype=np.float64)
    i = 0

    for image_path in image_paths:
        image = Image.open(image_path)
        image_np = np.array(image, dtype=np.float32)
        images_list.append(image_np)

        stamp_as_string = extract_timestamp(image_path)

        # Convert to uint64
        stamps[i] = np.float64(stamp_as_string)
        i = i + 1

    images_array = np.stack(images_list, axis=0)
    images_array = np.float32(images_array)
    return images_array,stamps

def main():
    cfg = get_config()
    fig, ax = plt.subplots()

    device = torch.device('cuda:'+str(cfg.cuda_id) if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print('Load data . . .')
    density_images,stamps = load_tiff_images_to_numpy(cfg.dataset_dir + '/bev_images')
    density_images = density_images[:, np.newaxis, :, :]
    print("Data shape:", density_images.shape)

    stamps = np.float64(stamps)

    # load model
    bev_sld_model = torch.load(cfg.network_path, map_location=torch.device('cpu')).to(device)
    bev_sld_model.eval()
        

    local_coords = create_local_coord_map(cfg.n_xy, cfg.grid_res)
    # add offset
    local_coords[0,:] += cfg.x_offset


    x_flattened = local_coords[0].flatten()  # Flatten x-channel
    y_flattened = local_coords[1].flatten()  # Flatten y-channel
    local_reshaped = np.column_stack((x_flattened, y_flattened,np.zeros(y_flattened.shape[0])))

    # init poses
    poses = None
    poses = np.zeros((density_images.shape[0],8))
    
    n_max = int( (cfg.n_div - cfg.n_padding)**2*cfg.lm_density)

    print("Start localization . . .")
    for i in tqdm(range(density_images.shape[0])): 

        input_tensor = torch.from_numpy(density_images[i:i+1,:,:,:]).float().to(device)

        with torch.no_grad():
            # execute model
            heat_map,corresp,coords  = bev_sld_model(input_tensor)
            heat_map, coords = heat_map[0,0,:,:].cpu().detach().numpy(), coords.cpu().detach().numpy()
            corresp = corresp[0,:,:,:].cpu().detach().numpy()

        xyz_local = np.zeros((n_max,3))
        xyz_scene = np.zeros((n_max,3))
        dists_lm = np.zeros((n_max,1))
        iter = 0

        if cfg.use_superpoint is True:
            peaks = pts[0:2,0:n_max].astype(int).T
        else:
            peaks = peak_local_max(heat_map,
                            min_distance=20,num_peaks=n_max)


        for j in range(peaks.shape[0]):
            row,col = peaks[j,0],peaks[j,1]
            
            # get local xy
            xy_local = np.array([local_coords[0,row,col],local_coords[1,row,col]])
            
            # map to corresp output dim
            row_class = int(round( float(corresp.shape[2]) * float(row) / float(cfg.n_xy) ))
            col_class = int(round( float(corresp.shape[2]) * float(col) / float(cfg.n_xy) ))

            # extract pixel correspondence scores
            pixel_corresp_scores = corresp[:,row_class,col_class]

            lm_id = np.argmax(pixel_corresp_scores)
            lm_id = np.min([lm_id,coords.shape[0]-1])  # ensure lm_id is within bounds
            val = pixel_corresp_scores[lm_id]

            xy_lm = coords[lm_id,:] 
            
            # save
            xyz_local[iter,:2] = xy_local[:]
            xyz_scene[iter,:2] = xy_lm[:]
            pixel_corresp_scores[lm_id] = 0
            iter = iter + 1

        # remove unused
        xyz_local = xyz_local[:iter,:]
        xyz_scene = xyz_scene[:iter,:]
        
        # estimate pose using ransac
        R,t = ransac_3d(xyz_local,xyz_scene,cfg.ransac_inlier_dist)
        
        if t.shape[0] == 1:
            t = t.reshape(3,1)

        quat = Rotation.from_matrix(R).as_quat()  # Returns [qx, qy, qz, qw]

        # Convert to Python array (list)
        quat_list = np.array(quat.tolist())
        newPose = np.hstack( (stamps[i:i+1,0:1],t.reshape(1, 3),quat_list[np.newaxis,:]))
        
        # save pose
        poses[i,:] = newPose
        
        ## plot
        ax = plt.gca()
        ax.cla() # clear things for fresh plot
    
        x_min = np.min(coords[:, 0]) - 5
        x_max = np.max(coords[:, 0]) + 5
        y_min = np.min(coords[:, 1]) - 5
        y_max = np.max(coords[:, 1]) + 5
        
        # plot landmarks
        for j in range(coords.shape[0]):
            x_global = coords[j,0]
            y_global = coords[j,1]
            circle = plt.Circle((x_global, y_global), 1.0, color='b',alpha=0.5,lw=0.5, fill=True)
            ax.add_patch(circle)
            
        # plot estimated position
        rot = R[:2,:2]
        x_global = t[0]
        y_global = t[1]
        end_point = rot @ np.array([10.0, 0.0]).T
        pos_arrow = plt.Arrow(x_global,y_global,end_point[0],end_point[1], width=6.0, color='red', label='Current Position')
        ax.add_patch(pos_arrow)
        
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)  # adjust if needed
        ax.axis('off')
    
        ax.set_aspect('equal')
        
        plt.draw()
        plt.pause(0.001)

    # create result dir
    result_dir = cfg.dataset_dir + '/results'+cfg.result_dir + '/'
    os.makedirs(result_dir, exist_ok=True)

    np.savetxt(result_dir + 'Poses.txt', poses, delimiter=' ', comments='', fmt='%.6f')
    print('Saved estimated poses to '+cfg.result_dir)


if __name__ == "__main__":
    main()