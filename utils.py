# import rosbag
# import sensor_msgs.point_cloud2 as pc2
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from tqdm import tqdm
import tifffile
import argparse
import yaml
import os
import random
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Affine2D
from scipy.linalg import logm, expm
from torchvision.transforms import RandomErasing
import torchvision.transforms as T

def get_config():
    parser = argparse.ArgumentParser(description='BEVL2Loc')

    parser.add_argument('--config', type=str, default='config/mcd_ntu_day_01_map.yaml', help='Directory of config file.')
    
    parser.add_argument('--eval_only', type=bool, help='Wheter this config is just for evaluation sequence without training.')
    parser.add_argument('--use_superpoint', type=bool, default=False, help='Wheter to use SuperPoint instead of self-supervised scene landmark detection.')
    parser.add_argument('--pose_file_dir', type=str, help='Directory of pose file in TUM format (.csv/.txt).')
    parser.add_argument('--pc_dir', type=str, help='Directory of point clouds in .pcd format.')
    parser.add_argument('--dataset_dir', type=str, help='Directory where bev images and labels are saved.')
    parser.add_argument('--keyframe_res', type=float,  help='Distance between consecutive keyframes in meters.')
    parser.add_argument('--grid_res', type=float, help='Grid resolution in meters.')
    parser.add_argument('--n_xy', type=int, help='Grid dimension in x and y direction.')
    parser.add_argument('--x_offset', type=float, help='X offset for the grid.')
    parser.add_argument('--bag_path', type=float, help='File path of the rosbag.')
    parser.add_argument('--pc_topic_name', type=float, help='Topic name of the point cloud2 in the rosbag.')
    

    # training
    parser.add_argument('--reduce_dataset_first_n', type=int, help='Reduce dataset in dataset creation to first n keyframes. Set to -1 to use all keyframes.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--test_frac', type=float, help='Fraction of the dataset to use for testing.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for detection training.')
    parser.add_argument('--n_div', type=int, help='Number of divisions for the grid for landmark detection loss.')
    parser.add_argument('--start_lr', type=float, help='Starting learning rate for detection training.')
    parser.add_argument('--network_path', type=str, help='Directory of the detection network.')

    parser.add_argument('--cuda_id', type=int, help='CUDA device ID to use for training.')

    parser.add_argument('--final_lr', type=float, help='Final learning rate for detection training.')
    parser.add_argument('--ransac_inlier_dist', type=float, help='Inlier distance for RANSAC in meters.')

    parser.add_argument('--n_padding', type=bool, help='Whether to pad the divided cells')
    parser.add_argument('--lm_density', type=bool, help='Fraction of landmarks are used.')

    
    parser.add_argument('--result_dir', type=str, help='Directory to save the results.')
    
    parser.add_argument('--loss_alpha', type=float, help='Loss weight for first term.')
    parser.add_argument('--loss_beta', type=float, help='Loss weight for second term.')
    parser.add_argument('--loss_gamma', type=float, help='Distance weight.')

    opt = parser.parse_args()

    # load yaml
    with open(opt.config) as f:
        yaml_cfg = yaml.safe_load(f)
    
    # replace params that are not in args
    if opt.pose_file_dir is None:
        opt.pose_file_dir = yaml_cfg['pose_file_dir']
        
    if opt.ransac_inlier_dist is None:
        opt.ransac_inlier_dist = yaml_cfg['ransac_inlier_dist']
        
    if opt.eval_only is None:
        opt.eval_only = yaml_cfg['eval_only']
        
    if opt.pc_dir is None:
        opt.pc_dir = yaml_cfg['pc_dir'] 

    if opt.dataset_dir is None:
        opt.dataset_dir = yaml_cfg['dataset_dir']

    if opt.keyframe_res is None:
        opt.keyframe_res = yaml_cfg['keyframe_res']
        
    if opt.grid_res is None:
        opt.grid_res = yaml_cfg['grid_res']

    if opt.n_xy is None:
        opt.n_xy = yaml_cfg['n_xy']

    if opt.x_offset is None:
        opt.x_offset = yaml_cfg['x_offset']

    if opt.bag_path is None:
        opt.bag_path = yaml_cfg['bag_path']

    if opt.pc_topic_name is None:
        opt.pc_topic_name = yaml_cfg['pc_topic_name']

    if opt.reduce_dataset_first_n is None:
        opt.reduce_dataset_first_n = yaml_cfg['reduce_dataset_first_n']
        
    if opt.network_path is None:
        opt.network_path = yaml_cfg['network_path']
        
    if opt.cuda_id is None:
        opt.cuda_id = yaml_cfg['cuda_id']
        
    if opt.result_dir is None:
        opt.result_dir = yaml_cfg['result_dir']
        
    if opt.lm_density is None:
        opt.lm_density = yaml_cfg['lm_density']
        
    if opt.n_padding is None:
        opt.n_padding = yaml_cfg['n_padding']
    
    if opt.n_div is None:   
        opt.n_div = yaml_cfg['n_div']
    
    # this section is relevant just for training sequences
    if opt.eval_only is False:
        
        if opt.loss_alpha is None:
            opt.loss_alpha = yaml_cfg['loss_alpha']
            
        if opt.loss_beta is None:
            opt.loss_beta = yaml_cfg['loss_beta']
        
        if opt.loss_gamma is None:
            opt.loss_gamma = yaml_cfg['loss_gamma']

        if opt.batch_size is None:
            opt.batch_size = yaml_cfg['batch_size']
    
        if opt.test_frac is None:
            opt.test_frac = yaml_cfg['test_frac']

        if opt.num_epochs is None:
            opt.num_epochs = yaml_cfg['num_epochs']
    
        if opt.start_lr is None:
            opt.start_lr = yaml_cfg['start_lr']

        if opt.final_lr is None:
            opt.final_lr = yaml_cfg['final_lr']
            
    return opt

def rotation_matrix_2d(theta):
    """
    Create a 2D rotation matrix for an angle theta (in radians).
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def save_config_as_yaml(cfg,file_name='config.yaml'):
    
    result_dir = cfg.dataset_dir + '/results'+cfg.result_dir + '/'
    
    # create folder
    os.makedirs(result_dir, exist_ok=True)
    
    # Convert Namespace to dict
    args_dict = vars(cfg)
    
    file_path = result_dir + file_name

    # Write to YAML
    with open(file_path, "w") as f:
        yaml.dump(args_dict, f)

def save_pcd_open3d(points: np.ndarray, filename: str, write_ascii: bool = True) -> None:
    """
    Save a 3xN numpy array of points to a .pcd file using Open3D.

    Args:
        points: np.ndarray of shape (3, N), where each column is [x, y, z].
        filename: Path to write the .pcd file to.
        write_ascii: If True, writes an ASCII PCD. Otherwise writes binary (smaller file).
    """
    # Validate shape
    if points.ndim != 2 or points.shape[0] != 3:
        raise ValueError(f"Expected points.shape == (3, N), got {points.shape}")

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    # Open3D expects an (N,3) array
    pcd.points = o3d.utility.Vector3dVector(points.T)

    # Write to file
    o3d.io.write_point_cloud(
        filename,
        pcd,
        write_ascii=write_ascii,
        compressed=False,       # you can set True for .pcd compression
        print_progress=True
    )

def read_poses(file_path: str) -> np.ndarray:
    """
    Read a poses file (CSV or space-delimited TXT), skipping the first line,
    and return an (N,8) NumPy array of [timestamp, x, y, z, qx, qy, qz, qw].
    
    Args:
        file_path: path to the .csv or .txt file
    
    Returns:
        data: np.ndarray of shape (N,8)
    """
    # Peek at the first line to choose delimiter
    with open(file_path, 'r') as f:
        first_line = f.readline()
    # Decide delimiter: comma if there's at least one ',', otherwise whitespace
    delim = ',' if ',' in first_line else None
    
    # Load the rest, skipping the first line
    data = np.loadtxt(file_path, delimiter=delim, skiprows=1)

    # skip first column (for mcd)
    if data.shape[1] > 8:
        data = data[:,1:]
    
    return data

def create_bev_density_image(pcd: np.ndarray, grid_res: float, grid_dim: tuple) -> np.ndarray:
    """
    Create a BEV density image from a point cloud.
    
    Args:
        pcd: Point cloud as a (3, N) numpy array.
        grid_res: Resolution of the grid in meters.
        grid_dim: Dimensions of the grid (height, width).
    
    Returns:
        bev_density_image: BEV density image as a numpy array.
    """
    # Initialize BEV density image
    bev_density_image = np.zeros(grid_dim, dtype=np.float32)

    # Convert point cloud to BEV coordinates
    x_indices = np.floor(pcd[0] / grid_res).astype(int)
    y_indices = np.floor(pcd[1] / grid_res).astype(int)

    ranges = np.sqrt(pcd[0]**2 + pcd[1]**2 + pcd[2]**2)

    x_indices = (x_indices+grid_dim[0]/2).astype(int)
    y_indices = (y_indices+grid_dim[1]/2).astype(int)

    valid = (x_indices > 0) & (y_indices > 0) & (x_indices < grid_dim[0]) & (y_indices < grid_dim[1])

    # Filter out points that are outside the grid
    x_indices = x_indices[valid]
    y_indices = y_indices[valid]
    ranges = ranges[valid]

    # Increment the density image at the corresponding indices
    for i in range(ranges.shape[0]):
        bev_density_image[x_indices[i], y_indices[i]] += 1 #ranges[i]

    # normalize the density image
    max_val = np.max(bev_density_image)
    med_val = np.median(bev_density_image[bev_density_image>0])

    if max_val > 0:
        bev_density_image /= max_val

    return bev_density_image


def create_local_coord_map(n_xy,grid_res):

    # Define the range and step size
    half_length = n_xy * grid_res / 2.0
    x_min, x_max = -half_length, half_length
    y_min, y_max = -half_length, half_length
    step_size = grid_res

    # Generate the coordinate grids
    x_values = np.linspace(x_min, x_max, num=n_xy)
    y_values = np.linspace(y_min, y_max, num=n_xy)

    # Create the meshgrid for x and y
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing='ij')

    # Create the numpy array with shape (2, 512, 512)
    result_array = np.stack([x_grid, y_grid], axis=0)

    # Display the shape to confirm
    return result_array


def save_array_as_tiff(array: np.ndarray, file_path: str) -> None:
    """
    Save a 2D NumPy array as a TIFF image.

    Parameters
    ----------
    array : np.ndarray
        A 2D array of shape (height, width). Can be any numeric dtype.
    file_path : str
        Output file path. Should end with .tif or .tiff.

    Raises
    ------
    ValueError
        If `array` is not 2-dimensional.
    """
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got array with shape {array.shape}")
    
    # tifffile will preserve the dtype (uint8, uint16, float32, etc.)
    tifffile.imwrite(file_path, array)



def extract_timestamp(path: str) -> str:
    """
    Extracts the numeric timestamp from the filename in the given path.
    Looks for the last sequence of digits, with optional decimal part,
    immediately before the file extension.
    
    Examples
    --------
    >>> extract_timestamp("/home/.../12345.5677.tif")
    '12345.5677'
    >>> extract_timestamp("/data/run_20210504_1591039200.jpg")
    '1591039200'
    """
    # 1. Get just the filename, e.g. "12345.5677.tif"
    filename = os.path.basename(path)
    # 2. Strip the extension -> "12345.5677"
    name, _ = os.path.splitext(filename)

    return name


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



class ErrorStatistics:
    def __init__(self, initial_capacity=1000):
        self.max_size = initial_capacity
        self.trans_errors = np.empty((self.max_size,), dtype=np.float64)
        self.rot_errors = np.empty((self.max_size,), dtype=np.float64)
        self.count = 0

    def _increase_capacity(self):
        new_size = self.max_size * 2
        self.trans_errors = np.resize(self.trans_errors, new_size)
        self.rot_errors = np.resize(self.rot_errors, new_size)
        self.max_size = new_size

    def add_element(self, query_R, query_t, ref_R, ref_t):
        """
        ref_R, query_R: 3x3 rotation matrices (numpy arrays)
        ref_t, query_t: 3x1 or 1x3 translation vectors (numpy arrays)
        """
        if self.count >= self.max_size:
            self._increase_capacity()

        # Translational error (Euclidean distance)
        t_err = np.linalg.norm(ref_t - query_t)

        # Rotational error (angle between rotation matrices)
        delta_R = np.linalg.inv(query_R[0:2,0:2]) @ ref_R[0:2,0:2]
        
        alpha = np.arctan2(delta_R[1, 0], delta_R[0, 0])

        r_err = np.abs(alpha)*180.0/np.pi 
        
        if r_err > 5:
            test = 1

        # Store
        self.trans_errors[self.count] = t_err
        self.rot_errors[self.count] = r_err
        self.count += 1
        
    def is_inlier(self, query_R, query_t, ref_R, ref_t):
        """
        ref_R, query_R: 3x3 rotation matrices (numpy arrays)
        ref_t, query_t: 3x1 or 1x3 translation vectors (numpy arrays)
        """
        if self.count >= self.max_size:
            self._increase_capacity()

        # Translational error (Euclidean distance)
        t_err = np.linalg.norm(ref_t - query_t)

        # Rotational error (angle between rotation matrices)
        delta_R = query_R[0:2,0:2].T @ ref_R[0:2,0:2]
        
        alpha = np.arctan2(delta_R[0, 1], delta_R[0, 0])

        r_err = np.abs(alpha)*180.0/np.pi 
        
        # success rate thresholds
        transl_threshold = 2.0
        rot_threshold = 5.0
        
        if t_err < transl_threshold and r_err < rot_threshold:
            return True
        else:
            return False


    def get_all_errors(self):
        return self.trans_errors[:self.count], self.rot_errors[:self.count]

    def get_mean_errors(self):
        return np.mean(self.trans_errors[:self.count]), np.mean(self.rot_errors[:self.count])
    
    def get_error_vectors(self):
        curr_t_err = self.trans_errors[:self.count]
        curr_r_err = self.rot_errors[:self.count]
        
        return curr_t_err, curr_r_err
    
    def get_statistics(self, latex_table_format=False,add_mean = False):
        
        curr_t_err = self.trans_errors[:self.count]
        curr_r_err = self.rot_errors[:self.count]
        
        # success rate thresholds
        transl_threshold = 2.0
        rot_threshold = 5.0
        
        # Calculate inliers based on thresholds
        inliers = (curr_t_err < transl_threshold) & (curr_r_err < rot_threshold)
        
        # calc success rate
        SR = 100.0*(np.sum(inliers) / self.count if self.count > 0 else 0.0)
        
        median_t = np.median(curr_t_err) if self.count > 0 else 0.0
        median_r = np.median(curr_r_err) if self.count > 0 else 0.0
        
        mean_t = np.mean(curr_t_err) if self.count > 0 else 0.0
        mean_r = np.mean(curr_r_err) if self.count > 0 else 0.0
        
        if add_mean is True:
            if latex_table_format:
                result_string = '{:.2f} & {:.2f} / {:.2f} & {:.2f} / {:.2f}'.format(
                    SR, median_t, mean_t, median_r, mean_r)
            else:
                result_string = 'SR: {:.2f}, Median T: {:.2f} m, Mean T: {:.2f} m, Median R: {:.2f} deg, Mean R: {:.2f} deg'.format(
                    SR, median_t, mean_t, median_r, mean_r)
                
        else:
            if latex_table_format:
                result_string = '{:.2f} & {:.2f} & {:.2f}'.format(
                    SR, median_t, median_r)
            else:
                result_string = 'SR: {:.2f}, Median T: {:.2f} m, Median R: {:.2f} deg'.format(
                    SR, median_t, median_r)
        
        return result_string
