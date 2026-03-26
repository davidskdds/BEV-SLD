from utils import read_poses, get_config, ErrorStatistics
import numpy as np
from scipy.spatial.transform import Rotation as R


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

