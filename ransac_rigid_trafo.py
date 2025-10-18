
import numpy as np
from typing import Tuple, Optional
import open3d

def compute_rigid_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the rigid transformation between two sets of corresponding 3D points.
   
    Args:
        A: (N,3) array of points from first point cloud
        B: (N,3) array of points from second point cloud
       
    Returns:
        R: (3,3) rotation matrix
        t: (3,) translation vector
    """
    # Center the points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    centered_A = A - centroid_A
    centered_B = B - centroid_B
   
    # Compute the covariance matrix and solve (overdetermined) equation system
    H = centered_A.T @ centered_B
    U, _, Vt = np.linalg.svd(H)
   
    # determine R
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
   
    # Determine t
    t = centroid_B - R @ centroid_A
   
    return R, t

def compute_distances(A: np.ndarray, B: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computes distances between transformed points A and points B.
   
    Args:
        A: (N,3) array of points from first point cloud
        B: (N,3) array of points from second point cloud
        R: (3,3) rotation matrix
        t: (3,) translation vector
       
    Returns:
        distances: (N,) array of distances between corresponding points
    """
    transformed_A = (R @ A.T).T + t
    distances = np.linalg.norm(transformed_A - B, axis=1)
    return distances

def ransac_3d(source_pc,target_pc, threshold=1.0):
    num_points = source_pc.shape[0]
    pred_t = np.zeros((1, 3))
    pred_q = np.zeros((1, 4))
    index1 = np.arange(0, num_points)
    index2 = np.arange(0, num_points)
    # np.random.shuffle(index1)
    index1 = np.expand_dims(index1, axis=1)
    index2 = np.expand_dims(index2, axis=1)
    corr = np.concatenate((index1, index2), axis=1)

    source_xyz = source_pc
    target_xyz = target_pc
    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_xyz)
    target.points = open3d.utility.Vector3dVector(target_xyz)
    corres = open3d.utility.Vector2iVector(corr)

    M = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source,
        target,
        corres,
        threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            open3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    pred_t[0, :] = M.transformation[:3, 3:].squeeze()
    # pred_q[0, :] = txq.mat2quat(M.transformation[:3, :3])

    R = M.transformation[:3, :3].copy()
    t = pred_t

    return R, t

def ransac_rigid_transform(A: np.ndarray, B: np.ndarray,
                         distance_threshold: float = 0.1,
                         max_iterations: int = 1000,
                         min_samples: int = 3,
                         stop_probability: float = 0.99) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Estimates rigid transformation between two point clouds using RANSAC.
   
    Args:
        A: (N,3) array of points from first point cloud
        B: (N,3) array of points from second point cloud
        distance_threshold: Maximum distance for a point to be considered an inlier
        max_iterations: Maximum number of RANSAC iterations
        min_samples: Minimum number of samples to compute transformation
        stop_probability: Probability of finding the correct transformation
       
    Returns:
        R: (3,3) rotation matrix
        t: (3,) translation vector
        num_inliers: Number of inliers for the best transformation
    """
    assert A.shape == B.shape, "Point clouds must have same shape"
    assert A.shape[1] == 3, "Points must be 3D"
   
    N = A.shape[0]
    best_R: Optional[np.ndarray] = None
    best_t: Optional[np.ndarray] = None
    best_num_inliers = 0
   
    # Adaptive number of iterations
    iterations = max_iterations
    best_inlier_ratio = 0
   
    for i in range(iterations):
        # Randomly sample minimum number of points
        idx = np.random.choice(N, min_samples, replace=False)
        sample_A = A[idx]
        sample_B = B[idx]
       
        # Compute transformation for samples
        R, t = compute_rigid_transform(sample_A, sample_B)
       
        # Compute distances for all points
        distances = compute_distances(A, B, R, t)
        inliers = distances < distance_threshold
        num_inliers = np.sum(inliers)
       
        # Update best result if we found more inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_R = R
            best_t = t
           
            # Update inlier ratio and number of iterations
            # inlier_ratio = num_inliers / N
            # if inlier_ratio > best_inlier_ratio:
            #     best_inlier_ratio = inlier_ratio
            #     num_samples = np.log(1 - stop_probability) / np.log(1 - inlier_ratio ** min_samples)
            #     iterations = min(int(num_samples), max_iterations)
       
    # Refine transformation using all inliers
    distances = compute_distances(A, B, best_R, best_t)
    inliers = distances < distance_threshold
    final_R, final_t = compute_rigid_transform(A[inliers], B[inliers])
   
    return final_R, final_t, best_num_inliers
