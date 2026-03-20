import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from ransac_rigid_trafo import ransac_3d
from scipy.spatial.transform import Rotation
from skimage.feature import peak_local_max
from tqdm import tqdm
from utils import get_config, create_local_coord_map, extract_timestamp
from matplotlib.patches import Patch, ConnectionPatch


def load_tiff_images_to_numpy(directory):
    image_paths = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith('.tif')
    ]
    image_paths.sort()

    images_list = []
    stamps = np.zeros((len(image_paths), 1), dtype=np.float64)

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        image_np = np.array(image, dtype=np.float32)
        images_list.append(image_np)

        stamp_as_string = extract_timestamp(image_path)
        stamps[i] = np.float64(stamp_as_string)

    images_array = np.stack(images_list, axis=0).astype(np.float32)
    return images_array, stamps


def normalize_to_01(img):
    img = img.astype(np.float32)
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - vmin) / (vmax - vmin)


def preprocess_density_for_plot(density_img):
    density_img = density_img.copy()

    if np.any(density_img > 0):
        non_zero_median = np.median(density_img[density_img > 0])
        density_img[density_img > 3.0 * non_zero_median] = 3.0 * non_zero_median

    if np.max(density_img) > 0:
        density_img = density_img / np.max(density_img)

    density_img = 1.0 - density_img
    return density_img


def compute_inlier_mask(xyz_local, xyz_scene, R, t, inlier_dist):
    if xyz_local.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    t = np.asarray(t).reshape(3)
    xyz_local_tf = (R @ xyz_local.T).T + t[None, :]
    residuals = np.linalg.norm(xyz_local_tf - xyz_scene, axis=1)
    inlier_mask = residuals < inlier_dist
    return inlier_mask


def plot_localization_result(
    fig,
    ax_left,
    ax_right,
    density_img,
    heat_map,
    coords,
    peaks,
    xyz_scene,
    inlier_mask,
    R,
    t,
    connection_artists,
    landmark_color='royalblue',
    title=None
):
    for artist in connection_artists:
        artist.remove()
    connection_artists.clear()

    ax_left.cla()
    ax_right.cla()

    density_img = preprocess_density_for_plot(density_img)
    heat_norm = normalize_to_01(heat_map)

    # LEFT: BEV + jet heatmap overlay
    ax_left.imshow(density_img, cmap='gray', origin='upper')
    ax_left.imshow(heat_norm, cmap='jet', alpha=0.35, origin='upper')

    ax_left.set_title("BEV + heatmap")
    ax_left.set_xlabel("col [px]")
    ax_left.set_ylabel("row [px]")

    if peaks.shape[0] > 0:
        ax_left.scatter(
            peaks[:, 1], peaks[:, 0],
            s=24,
            c='white',
            edgecolors='black',
            linewidths=0.7,
            zorder=5
        )

    if peaks.shape[0] > 0 and inlier_mask.shape[0] == peaks.shape[0]:
        inlier_peaks = peaks[inlier_mask]
        if inlier_peaks.shape[0] > 0:
            ax_left.scatter(
                inlier_peaks[:, 1], inlier_peaks[:, 0],
                s=42,
                facecolors='none',
                edgecolors='lime',
                linewidths=1.0,
                zorder=6
            )

    # RIGHT: landmarks + pose
    ax_right.set_title("Scene landmarks + estimated pose")
    ax_right.set_xlabel("x [m]")
    ax_right.set_ylabel("y [m]")

    ax_right.scatter(
    coords[:, 0], coords[:, 1],
    s=35,
    c=landmark_color,
    edgecolors='none',
    marker='o',
    zorder=2
    )

    if xyz_scene.shape[0] > 0 and inlier_mask.shape[0] == xyz_scene.shape[0]:
        xyz_scene_inliers = xyz_scene[inlier_mask]
        if xyz_scene_inliers.shape[0] > 0:
            ax_right.scatter(
                xyz_scene_inliers[:, 0], xyz_scene_inliers[:, 1],
                s=35,
                facecolors='none',
                edgecolors='lime',
                linewidths=1.0,
                marker='o',
                label='Inlier correspondences',
                zorder=3
            )

    x_global = float(t[0])
    y_global = float(t[1])
    heading_vec = R[:2, :2] @ np.array([8.0, 0.0], dtype=np.float32)

    ax_right.arrow(
        x_global, y_global,
        float(heading_vec[0]), float(heading_vec[1]),
        width=0.8,
        head_width=3.5,
        head_length=4.5,
        color='red',
        length_includes_head=True,
        zorder=5
    )

    ax_right.scatter(
        [x_global], [y_global],
        c='red',
        s=35,
        zorder=6
    )

    x_min = np.min(coords[:, 0]) - 5
    x_max = np.max(coords[:, 0]) + 5
    y_min = np.min(coords[:, 1]) - 5
    y_max = np.max(coords[:, 1]) + 5

    ax_right.set_xlim(x_min, x_max)
    ax_right.set_ylim(y_min, y_max)
    ax_right.set_aspect('equal')

    # only plot lines for RANSAC inliers
    if peaks.shape[0] > 0 and xyz_scene.shape[0] > 0 and inlier_mask.shape[0] == min(peaks.shape[0], xyz_scene.shape[0]):
        peaks_inliers = peaks[inlier_mask]
        xyz_scene_inliers = xyz_scene[inlier_mask]

        for j in range(peaks_inliers.shape[0]):
            row, col = peaks_inliers[j, 0], peaks_inliers[j, 1]
            x_lm, y_lm = xyz_scene_inliers[j, 0], xyz_scene_inliers[j, 1]

            con = ConnectionPatch(
                xyA=(col, row), coordsA=ax_left.transData,
                xyB=(x_lm, y_lm), coordsB=ax_right.transData,
                color='lime',
                linewidth=1.0,
                alpha=0.8
            )
            fig.add_artist(con)
            connection_artists.append(con)

    legend_elements = [
        Patch(facecolor=landmark_color, edgecolor='none', label='Scene landmarks'),
        Patch(facecolor='red', edgecolor='red', label='Estimated pose'),
        Patch(facecolor='lime', edgecolor='lime', label='RANSAC inliers'),
    ]
    ax_right.legend(handles=legend_elements, loc='upper right')

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def main():
    cfg = get_config()

    device = torch.device('cuda:' + str(cfg.cuda_id) if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print('Load data . . .')
    density_images, stamps = load_tiff_images_to_numpy(cfg.dataset_dir + '/bev_images')
    density_images = density_images[:, np.newaxis, :, :]
    print("Data shape:", density_images.shape)

    stamps = np.float64(stamps)

    bev_sld_model = torch.load(cfg.network_path, weights_only=False).to(device)
    bev_sld_model.eval()

    local_coords = create_local_coord_map(cfg.n_xy, cfg.grid_res)
    local_coords[0, :] += cfg.x_offset

    poses = np.zeros((density_images.shape[0], 8), dtype=np.float64)

    n_max = int((cfg.n_div - cfg.n_padding) ** 2 * cfg.lm_density)

    plt.ion()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 7))
    connection_artists = []

    print("Start localization . . .")
    for i in tqdm(range(density_images.shape[0])):

        input_tensor = torch.from_numpy(density_images[i:i+1, :, :, :]).float().to(device)

        with torch.no_grad():
            heat_map, corresp, coords = bev_sld_model(input_tensor)
            heat_map = heat_map[0, 0, :, :].cpu().detach().numpy()
            coords = coords.cpu().detach().numpy()
            corresp = corresp[0, :, :, :].cpu().detach().numpy()

        xyz_local = np.zeros((n_max, 3), dtype=np.float32)
        xyz_scene = np.zeros((n_max, 3), dtype=np.float32)
        iter_count = 0

        peaks = peak_local_max(
            heat_map,
            min_distance=20,
            num_peaks=n_max
        )

        for j in range(peaks.shape[0]):
            row, col = peaks[j, 0], peaks[j, 1]

            xy_local = np.array(
                [local_coords[0, row, col], local_coords[1, row, col]],
                dtype=np.float32
            )

            row_class = int(round(float(corresp.shape[1]) * float(row) / float(cfg.n_xy)))
            col_class = int(round(float(corresp.shape[2]) * float(col) / float(cfg.n_xy)))

            row_class = np.clip(row_class, 0, corresp.shape[1] - 1)
            col_class = np.clip(col_class, 0, corresp.shape[2] - 1)

            pixel_corresp_scores = corresp[:, row_class, col_class].copy()

            lm_id = int(np.argmax(pixel_corresp_scores))
            lm_id = min(lm_id, coords.shape[0] - 1)

            xy_lm = coords[lm_id, :]

            xyz_local[iter_count, :2] = xy_local
            xyz_scene[iter_count, :2] = xy_lm
            iter_count += 1

        xyz_local = xyz_local[:iter_count, :]
        xyz_scene = xyz_scene[:iter_count, :]
        peaks = peaks[:iter_count, :]

        R, t = ransac_3d(xyz_local, xyz_scene, cfg.ransac_inlier_dist)

        if t.ndim == 1:
            t = t.reshape(3, 1)
        elif t.shape[0] == 1:
            t = t.reshape(3, 1)

        inlier_mask = compute_inlier_mask(
            xyz_local=xyz_local,
            xyz_scene=xyz_scene,
            R=R,
            t=t,
            inlier_dist=cfg.ransac_inlier_dist
        )

        quat = Rotation.from_matrix(R).as_quat()
        quat_list = np.array(quat.tolist())

        new_pose = np.hstack((
            stamps[i:i+1, 0:1],
            t.reshape(1, 3),
            quat_list[np.newaxis, :]
        ))

        poses[i, :] = new_pose

        plot_localization_result(
            fig=fig,
            ax_left=ax_left,
            ax_right=ax_right,
            density_img=density_images[i, 0, :, :],
            heat_map=heat_map,
            coords=coords,
            peaks=peaks,
            xyz_scene=xyz_scene,
            inlier_mask=inlier_mask,
            R=R,
            t=t.reshape(3),
            connection_artists=connection_artists,
            landmark_color='royalblue',
            title=f"Global localization visualization - frame {i}"
        )

    result_dir = cfg.dataset_dir + '/results' + cfg.result_dir + '/'
    os.makedirs(result_dir, exist_ok=True)

    np.savetxt(result_dir + 'Poses.txt', poses, delimiter=' ', comments='', fmt='%.6f')
    print('Saved estimated poses to ' + cfg.result_dir)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()