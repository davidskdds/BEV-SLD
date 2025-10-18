from utils import read_poses,get_config,ErrorStatistics
import numpy as np
from scipy.spatial.transform import Rotation as R

def extract_z_rotation(q: np.ndarray, degrees: bool = False) -> float:
    """
    Extract the yaw angle (rotation about z-axis) from a quaternion [qx, qy, qz, qw].
    Returns the angle in radians by default, or in degrees if degrees=True.
    """
    # Create a Rotation object from the quaternion
    r = R.from_quat(q)
    # Extract Euler angles with the 'xyz' convention (where yaw is the third angle)
    _, _, yaw = r.as_euler('xyz', degrees=degrees)
    return yaw

def z_rotation_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Return a quaternion that represents only the rotation about the z-axis
    by extracting the yaw angle from the original quaternion.
    The new quaternion will have zero x and y components.
    """
    # First extract the yaw angle (in radians)
    yaw = extract_z_rotation(q, degrees=False)
    # Construct a new quaternion that represents only a rotation about the z-axis.
    # For a rotation about z by an angle θ, the quaternion is:
    # [0, 0, sin(θ/2), cos(θ/2)]
    qz = np.array([0.0,
                   0.0,
                   np.sin(yaw/2),
                   np.cos(yaw/2)])
    return qz

def quaternion_angle(q1: np.ndarray,
                     q2: np.ndarray,
                     degrees: bool = False) -> float:
    """
    Compute the angle between two quaternions q1, q2 (each as [qx, qy, qz, qw]).
    Returns the angle in radians by default, or in degrees if degrees=True.
    """
    # Normalize to ensure unit quaternions
    q1_u = q1 / np.linalg.norm(q1)
    q2_u = q2 / np.linalg.norm(q2)

    # Dot product
    dot = np.dot(q1_u, q2_u)

    # Clamp to avoid numerical errors outside [-1,1]
    dot = np.clip(dot, -1.0, 1.0)

    # Angle (radians)
    angle = 2.0 * np.arccos(abs(dot))

    if degrees:
        angle = np.degrees(angle)

    return angle



# load config
cfg = get_config()

results_dir = cfg.dataset_dir + '/results'+cfg.result_dir + '/'
poses_path = results_dir + 'Poses.txt'
ref_poses_path = cfg.pose_file_dir

curr_poses = read_poses(poses_path)
ref_poses = read_poses(ref_poses_path)

success_thresh_transl_m = 2.0
success_thresh_rot_deg = 5.0

# init
error_stats = ErrorStatistics()

for i in range(curr_poses.shape[0]):
    
    temporal_diff = np.abs(ref_poses[:,0]-curr_poses[i,0])
    corresp_ref_pose_id = np.argmin( temporal_diff )
    temporal_error = temporal_diff[corresp_ref_pose_id]

    # 2d translation error
    trans_diff = np.linalg.norm(curr_poses[i,1:3] - ref_poses[corresp_ref_pose_id,1:3])

    q1 = curr_poses[i,4:]
    q_ref = ref_poses[corresp_ref_pose_id,4:]
    
    # Create a Rotation object
    rot1 = R.from_quat(q1)  
    # Get the 3x3 matrix
    R1 = rot1.as_matrix()
    
    # Create a Rotation object
    rot_ref = R.from_quat(q_ref)  
    # Get the 3x3 matrix
    R_ref = rot_ref.as_matrix()
    

    error_stats.add_element(R1[0:2,0:2], curr_poses[i,1:3].flatten(), R_ref[0:2,0:2], ref_poses[corresp_ref_pose_id,1:3].flatten())


# get statistics
error_stats_str = error_stats.get_statistics(latex_table_format=False)
error_stats_latex_str = error_stats.get_statistics(latex_table_format=True)

print(error_stats_str)

text_file = open(results_dir + 'results.txt', "w")
text_file.write(error_stats_str+'\n\n'+error_stats_latex_str)
text_file.close()

