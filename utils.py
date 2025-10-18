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
import cv2
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

def read_poses_raw(file_path: str) -> np.ndarray:
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

    return data

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

    # Clip indices to fit within the grid dimensions
    # x_indices = np.clip(x_indices+grid_dim[0]/2, 0, grid_dim[0] - 1).astype(int)
    # y_indices = np.clip(y_indices+grid_dim[1]/2, 0, grid_dim[1] - 1).astype(int)

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

    # bev_density_image = np.log(bev_density_image + 1.0)  # Avoid log(0)

    return bev_density_image


def plot_numpy_array_as_image(array: np.ndarray):
    """
    Plot a 2D numpy array as an image using matplotlib.
    
    Args:
        array: 2D numpy array to plot.
    """
    import matplotlib.pyplot as plt

    plt.imshow(array)
    plt.axis('off')
    plt.show()


def create_local_coords(n_xy,grid_res):

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


def load_tiff_images_to_numpy_tifffile(directory,num_images=-1,index=-1):
    image_paths = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(('.tif', '.tiff'))
    )
    images = []

    i = 0
    for p in image_paths:

        if i == num_images:
            break

        if index >= 0 and i != index:
            i += 1
            continue

        try:
            arr = tifffile.imread(p).astype(np.float32)
        except Exception as e:
            print(f"⚠️  Failed to read {p} with tifffile: {e!r}")
            continue
        images.append(arr)

        i += 1

    if not images:
        raise RuntimeError("No valid TIFFs found!")

    return np.stack(images, axis=0)


# def transform_data(X, Y):
#     # settings
#     translation = 0.15
#     augm_prob = 1.0

#     if random.uniform(0.0, 1.0) > augm_prob:
#         # no augmentation
#         return X, Y

#     # ---- 1) roll in H (dim=1) ----
#     shift_frac_h = random.uniform(-translation, translation)
#     shift_h = int(X.shape[1] * shift_frac_h)
#     X = torch.roll(X, shifts=shift_h, dims=1)
#     Y = torch.roll(Y, shifts=shift_h, dims=1)
#     if shift_h > 0:
#         # positive shift: rows [0:shift_h] are wrapped from bottom → zero them
#         X[:, :shift_h, :] = 0
#         Y[:, :shift_h, :] = 0
#     elif shift_h < 0:
#         # negative shift: rows [shift_h:] (i.e. last |shift_h|) wrapped from top → zero them
#         X[:, shift_h:, :] = 0
#         Y[:, shift_h:, :] = 0

#     # ---- 2) roll in W (dim=2) ----
#     shift_frac_w = random.uniform(-translation, translation)
#     shift_w = int(X.shape[2] * shift_frac_w)
#     X = torch.roll(X, shifts=shift_w, dims=2)
#     Y = torch.roll(Y, shifts=shift_w, dims=2)
#     if shift_w > 0:
#         X[:, :, :shift_w] = 0
#         Y[:, :, :shift_w] = 0
#     elif shift_w < 0:
#         X[:, :, shift_w:] = 0
#         Y[:, :, shift_w:] = 0

#     # ---- 3) rotation ----
#     angle = random.uniform(0, 360)
#     X = TF.rotate(X, angle, interpolation=TF.InterpolationMode.BILINEAR)
#     Y = TF.rotate(Y, angle, interpolation=TF.InterpolationMode.BILINEAR)

#     # ---- 4) brightness scaling ----
#     X = random.uniform(0.5, 1.5) * X

#     return X, Y

def transform_dataOld(X,Y):
    
    # settings
    translation = 0.15
    augm_prob = 1.0
    
    if random.uniform(0.0, 1.0) > augm_prob:
        # no augmentation
        return X, Y
    
    
    x_shape = X.shape
    y_shape = Y.shape
    
    # augment
    shift_dim2 = random.uniform(-translation, translation)
    X = torch.roll(X, shifts=int(X.shape[1]*shift_dim2), dims = 1)
    Y = torch.roll(Y, shifts=int(Y.shape[1]*shift_dim2), dims = 1)

    shift_dim3 = random.uniform(-translation, translation)
    X = torch.roll(X, shifts=int(X.shape[2]*shift_dim3), dims = 2)
    Y = torch.roll(Y, shifts=int(Y.shape[2]*shift_dim3), dims = 2)

    angle = random.uniform(0, 360)

    X = TF.rotate(X, angle,interpolation=TF.InterpolationMode.BILINEAR)
    Y = TF.rotate(Y, angle,interpolation=TF.InterpolationMode.BILINEAR)
    
    # multiply with random factor
    X = random.uniform(0.5, 1.5) * X


    return X,Y

def random_affine_single(X, Y, max_angle=30, max_trans=0.2, mode='bilinear'):
    """
    Apply the same random rotation + translation to one image X and one label/mask Y.
    
    Args:
      X: Tensor[C, H, W]
      Y: Tensor[C, H, W]
      max_angle: max rotation in degrees (±)
      max_trans: max translation as fraction of dims (±)
      mode: 'bilinear' for images, 'nearest' for masks if needed
      
    Returns:
      X_t, Y_t: Transformed tensors of shape [C, H, W]
    """
    # 1) Turn into a “batch” of 1
    Xb = X.unsqueeze(0)  # [1, C, H, W]
    Yb = Y.unsqueeze(0)

    # 2) Sample one angle and translations
    angle = (torch.rand(1) * 2 - 1) * max_angle        # degrees
    tx = (torch.rand(1) * 2 - 1) * max_trans           # fraction of width
    ty = (torch.rand(1) * 2 - 1) * max_trans           # fraction of height
    rad = angle * torch.pi / 180.0
    c, s = torch.cos(rad), torch.sin(rad)

    # 3) Build the 2×3 theta matrix
    theta = torch.tensor([[
        [ c, -s, tx],
        [ s,  c, ty]
    ]], dtype=X.dtype, device=X.device)  # shape [1,2,3]

    # 4) Make a sampling grid for that batch
    grid = F.affine_grid(theta, Xb.size(), align_corners=True)

    # 5) Warp both
    Xb_t = F.grid_sample(Xb, grid, mode=mode, padding_mode='zeros', align_corners=True)
    Yb_t = F.grid_sample(Yb, grid, mode=mode, padding_mode='border', align_corners=True)

    # 6) Remove batch dim
    return Xb_t.squeeze(0), Yb_t.squeeze(0)

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


def get_median_nearest_neighbor_dist(coords):

    tree = cKDTree(coords)

    # Query nearest neighbor (k=2 because the first result is the point itself)
    distances, indices = tree.query(coords, k=2)

    # distances[:, 1] gives the distance to the nearest *other* point
    nearest_distances = distances[:, 1]

    median_dist = np.median(nearest_distances)

    return median_dist

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def load_and_stack_npy(dir_path,reduce_to=-1):
    """
    Loads all .npy files from the given directory, sorted by filename, 
    and stacks them into a single NumPy array of shape (n_total, n, n, k).

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the .npy files. Each .npy file 
        should contain an array of shape (n, n, k).

    Returns
    -------
    stacked_array : numpy.ndarray
        A NumPy array of shape (n_total, n, n, k), where n_total is the 
        number of .npy files in the directory. Files are stacked in 
        ascending filename order.
    """
    # List all files ending with .npy
    filenames = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    if not filenames:
        raise ValueError(f"No .npy files found in directory: {dir_path}")

    # Sort filenames lexicographically
    filenames.sort()
    
    if reduce_to > 0:
        # Reduce to first `reduce_to` files
        filenames = filenames[:reduce_to]

    # Load each file and collect into a list
    arrays = []
    for fname in filenames:
        full_path = os.path.join(dir_path, fname)
        arr = np.load(full_path)  # expecting shape (n, n, k)
        arrays.append(arr)

    # Stack along a new first axis: result shape will be (n_total, n, n, k)
    stacked_array = np.stack(arrays, axis=0)
    return stacked_array

def get_axis_angle(R):
    # 1) compute the matrix logarithm
    L = logm(R)
    # 2) extract the (x,y,z) components from the skew matrix
    #    here L = [  0  -wz  wy
    #               wz    0  -wx
    #              -wy  wx    0 ]
    r = np.zeros(3)
    r[0] =  L[2,1]
    r[1] = -L[2,0]
    r[2] =  L[1,0]
    
    return r

def rotm2axangle(R):
    # 1) compute the matrix logarithm
    L = logm(R)
    # 2) extract the (x,y,z) components from the skew matrix
    #    here L = [  0  -wz  wy
    #               wz    0  -wx
    #              -wy  wx    0 ]
    wx =  L[2,1]
    wy = -L[2,0]
    wz =  L[1,0]
    
    r = np.zeros(3)
    r[0] = wx
    r[1] = wy
    r[2] = wz
    
    return r

def extract_partial_rotation(R, use_x=True, use_y=True, use_z=True):
    # 1) compute the matrix logarithm
    L = logm(R)
    # 2) extract the (x,y,z) components from the skew matrix
    #    here L = [  0  -wz  wy
    #               wz    0  -wx
    #              -wy  wx    0 ]
    wx =  L[2,1]
    wy = -L[2,0]
    wz =  L[1,0]
    
    # set rot components to zero if not used
    if not use_x:
        wx = 0.0
        
    if not use_y:
        wy = 0.0
        
    if not use_z:
        wz = 0.0
        
    if np.linalg.norm( np.array([wx,wy,wz]) ) < 1e-4:
        return np.identity(3)
    
    # 4) rebuild the skew
    L_rp = np.array([[  0, -wz,  wy],
                     [ wz,   0, -wx],
                     [-wy,  wx,   0]])
    # 5) exponentiate back into SO(3)
    return expm(L_rp)

def create_heatmap_im(grayscale_img, error_img, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlay an error map onto a grayscale image, add a color scale bar, and display it using OpenCV.
    
    :param grayscale_img: NumPy array, grayscale image (2D array).
    :param error_img: NumPy array, error values per pixel (2D array).
    :param alpha: Float, blending factor for overlay (default 0.6).
    :param colormap: OpenCV colormap to use for errors (default cv2.COLORMAP_JET).
    :return: Blended image with a scale bar.
    """
    # Normalize grayscale image to 0-255
    grayscale_img = cv2.normalize(grayscale_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grayscale_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
    
    # Normalize error image to 0-255
    error_min, error_max = np.min(error_img), np.max(error_img)
    error_img = cv2.normalize(error_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if np.max(error_img)-np.min(error_img) < 1e-4:
        # If the error image is constant, return the grayscale image
        return grayscale_img
    
    # Apply colormap
    error_colored = cv2.applyColorMap(error_img, colormap)

    # Blend images
    blended = cv2.addWeighted(grayscale_img, 1 - alpha, error_colored, alpha, 0)

    return blended  # Return the final image with the scale bar

def plot_image_at(ax, img, x, y, angle=0, width=None, height=None,alpha_in=1.0):
    """
    Draw `img` on axes `ax`, centered at data‐coords (x, y), rotated by `angle` degrees.
    Optionally specify `width`/`height` in data units (otherwise uses pixel size).
    """
    # image size in data units
    h_px, w_px = img.shape[:2]
    if width is None and height is None:
        width, height = w_px, h_px
    elif width is None:
        width = height * (w_px / h_px)
    elif height is None:
        height = width * (h_px / w_px)

    # compute the extent so that the image is centered at (x, y)
    extent = [
        x - width/2,  # left
        x + width/2,  # right
        y - height/2, # bottom
        y + height/2  # top
    ]

    # build a rotation about (x, y) in data coordinates
    trans = Affine2D().rotate_deg_around(x, y, angle) + ax.transData

    ax.imshow(img, extent=extent, origin='upper', transform=trans,alpha= alpha_in)

def create_image(ax,landmarks, density_im,lm_out, local_coords, scene_coords, rot, translation,\
                 cfg,save_im=False, clear_fig=True,set_ax_lim=True,plot_pose=True,show_corresp = True,show_movement=False,show_local_pts=False,show_all_lms=True,name_attach=''):


    # now make a circle with no fill, which is good for hi-lighting key results
    if clear_fig:
        ax = plt.gca()
        ax.cla() # clear things for fresh plot

    # change default range so that new circles will work
    if set_ax_lim:
        edge_dist = 20.0
        x_min = np.min(landmarks[:,0]) - edge_dist
        x_max = np.max(landmarks[:,0]) + edge_dist
        
        y_min = np.min(landmarks[:,1]) - edge_dist
        y_max = np.max(landmarks[:,1]) + edge_dist
        
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
    
    # if show_movement is not True:
    #     for i in range(landmarks.shape[0]):
    #         # circle = plt.Circle((landmarks[i,0], landmarks[i,1]), 2.5, color='yellow',lw=0.5, fill=False)
    #         circle = plt.Circle((landmarks[i,0], landmarks[i,1]), 2.5, color='w',lw=0.5, fill=False)
    #         ax.add_patch(circle)
    
    # add inlier circles
    local_scene = (rot @ local_coords.T + translation).T
    
    diff = np.sum((local_scene - scene_coords)**2,1)
    
    valid_lms = diff < 3.0
    
    for i in range(landmarks.shape[0]):
        circle = plt.Circle((landmarks[i,0], landmarks[i,1]), 2.5, color='w',lw=0.5, fill=False)
        ax.add_patch(circle)
    
    valid_sc = scene_coords[valid_lms,:]
    if show_corresp:
        if valid_sc.shape[0] > 0:
            for i in range(valid_sc.shape[0]):
                # circle = plt.Circle((landmarks[i,0], landmarks[i,1]), 2.5, color='b',lw=0.5, fill=False)
                # ax.add_patch(circle)
            
                circle = plt.Circle((valid_sc[i,0], valid_sc[i,1]), 2.5, color='limegreen',lw=0.5, fill=False)
                ax.add_patch(circle)
                
                # create line
                line = ConnectionPatch(xyA=translation[0:2,:], xyB=valid_sc[i,:],
                        coordsA='data', coordsB='data',
                        axesA=ax, axesB=ax,
                        color='silver', linestyle='--', linewidth=0.3)
                
                ax.add_patch(line)
    
        
    # add current pos as arrow
    if plot_pose:
        end_point = rot @ np.array([10.0, 0.0]).T
        pos_arrow = plt.Arrow(translation[0,0],translation[1,0],end_point[0],end_point[1], width=6.0, color='lime', label='Current Position')
        ax.add_patch(pos_arrow)
    
    # inverted_im = np.ones_like(density_im) - density_im
    
    # lm_out = np.log(1.0+lm_out)
    
    # lm_out = lm_out / np.max(lm_out) # normalize to 0-1
    
    # create image
    local_im = create_heatmap_im(density_im,lm_out,0.7) # former 1
    # local_im = create_heatmap_im(inverted_im,lm_out,0.0)
    
    # reorder
    local_im = local_im[:, :, [2, 1, 0]]
    
    im_size = cfg.n_xy * cfg.grid_res
    theta = np.arctan2(rot[1, 0], rot[0, 0])
    theta = np.degrees(theta)
    
    # label
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    
    ax.axis('off')
    
    plot_image_at(ax, local_im, translation[0], translation[1], angle=theta+90, width=im_size, height=im_size,alpha_in=0.7)
    
    if show_local_pts is True:
        for i in range(local_coords.shape[0]):
                circle = plt.Circle((local_coords[i,0], local_coords[i,1]), 1.0, color='orange', fill=True)
                ax.add_patch(circle)
    
    if show_movement:
        for i in range(local_coords.shape[0]):
            circle = plt.Circle((local_coords[i,0], local_coords[i,1]), 1.0, color='orange', fill=True)
            ax.add_patch(circle)
            
            circle = plt.Circle((scene_coords[i,0], scene_coords[i,1]), 1.0, color='lime', fill=False)
            ax.add_patch(circle)
            
            line = ConnectionPatch(xyA=local_coords[i,:], xyB=scene_coords[i,:],
                            coordsA='data', coordsB='data',
                            axesA=ax, axesB=ax,
                            color='orange', linestyle='--', linewidth=1.0)
            
            ax.add_patch(line)
        
    if save_im:
        plt.savefig(cfg.dataset_dir+'/local_im2'+name_attach+'.png', dpi=1000, pad_inches = 0,bbox_inches='tight')
    plt.draw()
    plt.pause(0.1)
    # plt.show()
    
    # fig.savefig('plotcircles2.png')
    
def create_init_landmarks(poses_tum, cfg):

    n_local = int(cfg.n_xy / cfg.n_div)
    cell_res = cfg.n_xy/cfg.n_div * cfg.grid_res

    local_coords = create_local_coords(cfg.n_div,cell_res)
    
    # padding
    local_coords = local_coords[:,cfg.n_padding:cfg.n_div-cfg.n_padding,cfg.n_padding:cfg.n_div-cfg.n_padding]

    x_local = local_coords[0,:].flatten()
    y_local = local_coords[1,:].flatten()

    seed_points = np.zeros((poses_tum.shape[0]*cfg.n_div**2,2))
    
    div_min_padding = cfg.n_div - 2*cfg.n_padding

    for i in range(poses_tum.shape[0]):

        seed_points[i*div_min_padding**2:(i+1)*div_min_padding**2,0] = poses_tum[i,1] + x_local
        seed_points[i*div_min_padding**2:(i+1)*div_min_padding**2,1] = poses_tum[i,2] + y_local

    # grid downsampling
    points_ds = downsample_grid_average(seed_points,cell_res)
    
    # random selection
    points_ds = points_ds[np.random.rand(points_ds.shape[0]) < cfg.lm_density , :]
    

    return points_ds

def downsample_grid_average(points, grid_size):
    """
    Downsample a set of 2D points by replacing all points in each grid cell
    with their centroid (average).

    Parameters:
        points (np.ndarray): Array of shape (n, 2) containing 2D points.
        grid_size (float): Size of each square grid cell. Must be > 0.

    Returns:
        np.ndarray: Downsampled points array of shape (m, 2), where m is the
                    number of non-empty cells. Each row is the average (x, y)
                    of all original points falling into that cell.
    """
    # ————————————
    # 1. Basic sanity checks
    # ————————————
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("`points` must be an (n, 2) array.")
    if points.size == 0:
        return points.copy()    # nothing to do if empty
    if grid_size <= 0:
        raise ValueError("`grid_size` must be a positive number.")

    # ————————————————
    # 2. Shift so min‐xy → (0,0), compute cell indices
    # ————————————————
    # (This ensures that cells start counting from zero regardless of
    #  whether points have negative coords or not.)
    min_xy = np.min(points, axis=0)                      # shape (2,)
    shifted = points - min_xy                            # shape (n,2)
    cell_indices = np.floor(shifted / grid_size).astype(int)
    # Now cell_indices[i] = integer pair (i_cell, j_cell) for points[i]

    # ————————————————————————————
    # 3. Accumulate sums and counts per (i_cell, j_cell) key
    # ————————————————————————————
    # We'll build a dict:  key → [sum_x, sum_y, count]
    sums_and_counts = {}
    for pt, (ci, cj) in zip(points, cell_indices):
        key = (int(ci), int(cj))
        if key not in sums_and_counts:
            sums_and_counts[key] = [pt[0], pt[1], 1]
        else:
            sums_and_counts[key][0] += pt[0]
            sums_and_counts[key][1] += pt[1]
            sums_and_counts[key][2] += 1

    # —————————————————————————————
    # 4. Form the centroid of each occupied cell
    # —————————————————————————————
    averaged_pts = []
    for (ci, cj), (sum_x, sum_y, cnt) in sums_and_counts.items():
        averaged_pts.append([sum_x / cnt, sum_y / cnt])

    return np.array(averaged_pts)


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
