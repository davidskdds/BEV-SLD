
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from tqdm import tqdm

from utils import read_poses,create_bev_density_image,create_local_coord_map,save_array_as_tiff,get_config
import os
import shutil



def main():
    # settings
    cfg = get_config()

    # create tuple for grid dimensions
    grid_dim = (cfg.n_xy,cfg.n_xy)

    # create directories
    bev_img_dir = cfg.dataset_dir+'/bev_images/'
    label_x_dir = cfg.dataset_dir+'/label_x/'
    label_y_dir = cfg.dataset_dir+'/label_y/'

    # remove old data
    shutil.rmtree(bev_img_dir, ignore_errors=True)
    shutil.rmtree(label_x_dir, ignore_errors=True)
    shutil.rmtree(label_y_dir, ignore_errors=True)

    # create directories if they do not exist
    os.makedirs(bev_img_dir, exist_ok=True)
    
    if cfg.eval_only is False:
        os.makedirs(label_x_dir, exist_ok=True)
        os.makedirs(label_y_dir, exist_ok=True)

    # read poses
    poses_tum = read_poses(cfg.pose_file_dir)
    first_stamp = poses_tum[0,0]
    last_stamp = poses_tum[poses_tum.shape[0]-1,0]

    # init
    last_key_pos = -9999.9*np.ones(3)
    sample_num = 0

    # collect and sort all .pcd filenames by their leading timestamp
    pcd_files = sorted(
        [f for f in os.listdir(cfg.pc_dir) if f.endswith('.pcd')],
        key=lambda fn: float(fn.split('_')[0].replace('.pcd', ''))
    )

    # Iterate over sorted pcd files
    for filename in tqdm(pcd_files):
        pcd_path  = os.path.join(cfg.pc_dir, filename)

        # extract stamp from filename
        stamp_sec = float(filename.split('_')[0].replace('.pcd', ''))

        # prepare deskewing
        delta_stamps = np.abs(poses_tum[:,0]-stamp_sec)
        curr_pose_id = np.argmin(delta_stamps)

        if delta_stamps[curr_pose_id] > 0.2:
            print('Warning: Minimum delta of pc2-msg and posefile poses timestamps is: '+str(delta_stamps[curr_pose_id]))


        # extract pose params
        t0 = poses_tum[curr_pose_id,1:4]
        q0 = poses_tum[curr_pose_id,4:]

        # Create a Rotation object
        rot0 = R.from_quat(q0)  
        # Get the 3x3 matrix
        R0 = rot0.as_matrix()
        
        # skip
        if np.linalg.norm(last_key_pos-t0) < cfg.keyframe_res:
            continue
        else:
            last_key_pos = t0
        
        # Load the pcd file
        pcd = o3d.io.read_point_cloud(pcd_path)

        # grid downsampling using o3d
        pcd = pcd.voxel_down_sample(voxel_size=cfg.grid_res/2.0)

        xyz_load = np.asarray(pcd.points).T
        
        # add x offset
        xyz_load[0,:] += cfg.x_offset
        
        image = create_bev_density_image(xyz_load, cfg.grid_res, grid_dim)

        # save image
        save_array_as_tiff(image,bev_img_dir+str(stamp_sec)+'.tif')

        if cfg.eval_only is False:
            # create labels
            local_xy = create_local_coord_map(grid_dim[0],cfg.grid_res)

            # add offset
            local_xy[0,:] += cfg.x_offset

            # reshape 2x512x512 to 2x-1
            local_xy = local_xy.reshape(2,-1)

            # apply transform
            global_xy = R0[0:2,0:2]@local_xy + t0[0:2].reshape(2,1)

            # reshape to 512x512
            global_xy = global_xy.reshape(2,grid_dim[0],grid_dim[1])

            # save labels
            save_array_as_tiff(global_xy[0,:,:],label_x_dir+str(stamp_sec)+'.tif')
            save_array_as_tiff(global_xy[1,:,:],label_y_dir+str(stamp_sec)+'.tif')
            
        sample_num += 1

        if sample_num == cfg.reduce_dataset_first_n:
            break


if __name__ == "__main__":
    main()