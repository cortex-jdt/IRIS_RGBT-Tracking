# Full Modified Script for Calibration and Validation
# NEW WORKFLOW: Undistorts the RGB and Depth images first, then warps the
# thermal image directly into the clean, undistorted RGB space.
# This is a more intuitive workflow that produces a geometrically correct dataset.

import yaml
import numpy as np
import os
import cv2
import re
from scipy.spatial.transform import Rotation

def parse_yaml_file(filepath):
    """
    Loads a YAML file, skipping the first header line, and creates a map
    using the common timestamp as the key.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        try:
            lines = f.readlines()
            if lines and not ':' in lines[0]:
                    yaml_content = "".join(lines[1:])
            else:
                    yaml_content = "".join(lines)
            data = yaml.safe_load(yaml_content)
            extrinsics_map = {}
            timestamp_pattern = re.compile(r"(\d{8}_\d{6}_\d{9})")
            for entry in data.get('extrinsics', []):
                base_filename = entry['image'].split('/')[-1].split('\\')[-1]
                match = timestamp_pattern.search(base_filename)
                if match:
                    timestamp_key = match.group(1)
                    extrinsics_map[timestamp_key] = {
                        'rotation': np.array(entry['rotation']),
                        'translation': np.array(entry['translation'])
                    }
                else:
                    print(f"Warning: Could not extract timestamp key from filename: {base_filename}")
            data['extrinsics_map'] = extrinsics_map
            data['intrinsic'] = {k: v for k, v in data['intrinsic'].items()}
            return data
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {filepath}: {e}")
            return None

def create_transform_matrix(R, t):
    """Creates a 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def average_rotations(rotation_matrices):
    """Averages multiple rotation matrices using quaternion averaging."""
    quaternions = [Rotation.from_matrix(R).as_quat() for R in rotation_matrices]
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[0], quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]
    avg_quat = np.mean(quaternions, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    return Rotation.from_quat(avg_quat).as_matrix()

def save_extrinsics_to_yaml(output_dir, filename, R, t, from_cam, to_cam):
    """Saves the final camera-to-camera extrinsics to a YAML file."""
    filepath = os.path.join(output_dir, filename)
    data_to_save = {
        f'R_{from_cam}_to_{to_cam}': R.tolist(),
        f'T_{from_cam}_to_{to_cam}': t.tolist()
    }
    with open(filepath, 'w') as f:
        yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False)
    print(f"Successfully saved final averaged extrinsics to {filepath}")

def calculate_and_average_extrinsics(cam1_data, cam2_data, cam1_name, cam2_name):
    """Calculates camera-to-camera extrinsics for all common views and averages them."""
    print(f"\n--- Processing Pair: {cam1_name} and {cam2_name} ---")
    cam1_images = set(cam1_data['extrinsics_map'].keys())
    cam2_images = set(cam2_data['extrinsics_map'].keys())
    common_images = sorted(list(cam1_images.intersection(cam2_images)))
    if not common_images:
        print(f"Error: No common calibrated images found between the two cameras for {cam1_name} and {cam2_name}.")
        return None, None, None
    print(f"Found {len(common_images)} common images for averaging.")
    all_translations, all_rotations = [], []
    for img_key in common_images:
        R_board_to_cam1 = cam1_data['extrinsics_map'][img_key]['rotation']
        t_board_to_cam1 = cam1_data['extrinsics_map'][img_key]['translation']
        R_board_to_cam2 = cam2_data['extrinsics_map'][img_key]['rotation']
        t_board_to_cam2 = cam2_data['extrinsics_map'][img_key]['translation']
        T_board_to_cam1 = create_transform_matrix(R_board_to_cam1, t_board_to_cam1)
        T_board_to_cam2 = create_transform_matrix(R_board_to_cam2, t_board_to_cam2)
        T_cam2_to_board = np.linalg.inv(T_board_to_cam2)
        T_cam2_to_cam1 = T_board_to_cam1 @ T_cam2_to_board
        all_rotations.append(T_cam2_to_cam1[:3, :3])
        all_translations.append(T_cam2_to_cam1[:3, 3])
    avg_translation = np.mean(all_translations, axis=0)
    avg_rotation = average_rotations(all_rotations)
    return avg_rotation, avg_translation, common_images

def warp_thermal_to_rgb(thermal_img, depth_img, K_rgb, D_rgb, K_thermal, D_thermal, R_thermal_to_rgb, t_thermal_to_rgb, depth_scale):
    """
    Warp a thermal image to a given RGB viewpoint.
    The RGB viewpoint can be distorted or undistorted, depending on the D_rgb passed.
    """
    t_thermal_to_rgb_meters = t_thermal_to_rgb / 100.0
    R_rgb_to_thermal = R_thermal_to_rgb.T
    t_rgb_to_thermal = -R_rgb_to_thermal @ t_thermal_to_rgb_meters
    h, w = depth_img.shape[:2]

    # Create a grid of pixel coordinates in the target RGB image
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    rgb_pixels = np.vstack((x_coords.ravel(), y_coords.ravel())).T.astype(np.float32)

    # Correct the RGB pixel locations for distortion to get ideal coordinates
    # If D_rgb is all zeros (for an undistorted image), this step just prepares coordinates for 3D math
    rgb_pixels_undistorted = cv2.undistortPoints(rgb_pixels, K_rgb, D_rgb, P=K_rgb)
    rgb_pixels_undistorted_hom = np.vstack((rgb_pixels_undistorted.reshape(-1, 2).T, np.ones(w*h)))

    depth_values = depth_img.ravel() * depth_scale
    has_depth_mask = depth_values > 0
    no_depth_mask = ~has_depth_mask
    projected_pixels_thermal = np.zeros((w * h, 2), dtype=np.float32)

    # Process pixels WITH depth
    if np.any(has_depth_mask):
        rgb_coords_with_depth = rgb_pixels_undistorted_hom[:, has_depth_mask]
        depth_vals_with_depth = depth_values[has_depth_mask]
        points_3d_rgb = np.linalg.inv(K_rgb) @ rgb_coords_with_depth * depth_vals_with_depth
        points_3d_thermal = R_rgb_to_thermal @ points_3d_rgb + t_rgb_to_thermal.reshape(3, 1)
        projected, _ = cv2.projectPoints(points_3d_thermal.T, np.zeros(3), np.zeros(3), K_thermal, D_thermal)
        projected_pixels_thermal[has_depth_mask] = projected.reshape(-1, 2)

    # Process pixels WITHOUT depth
    if np.any(no_depth_mask):
        rgb_coords_no_depth = rgb_pixels_undistorted_hom[:, no_depth_mask]
        normalized_rgb_coords = np.linalg.inv(K_rgb) @ rgb_coords_no_depth
        rotated_thermal_coords = R_rgb_to_thermal @ normalized_rgb_coords
        projected, _ = cv2.projectPoints(rotated_thermal_coords.T, np.zeros(3), np.zeros(3), K_thermal, D_thermal)
        projected_pixels_thermal[no_depth_mask] = projected.reshape(-1, 2)

    map_x = projected_pixels_thermal[:, 0].reshape(h, w).astype(np.float32)
    map_y = projected_pixels_thermal[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(thermal_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def visualize_and_save_all(base_path, output_dir, R_left_to_rgb, t_left_to_rgb, R_right_to_rgb, t_right_to_rgb, rgb_data, left_data, right_data, common_images):
    """
    MODIFIED: Aligns images by first undistorting the RGB/Depth data, then
    warping the thermal image directly into the undistorted RGB space.
    """
    print("\n--- Visualizing and Saving Final Alignment (Undistort-First Workflow) ---")

    K_rgb = np.array([[rgb_data['intrinsic']['fx'], 0, rgb_data['intrinsic']['cx']], [0, rgb_data['intrinsic']['fy'], rgb_data['intrinsic']['cy']], [0, 0, 1]])
    D_rgb = np.array([rgb_data['intrinsic'].get(k, 0) for k in ['k1', 'k2', 'p1', 'p2', 'k3']])
    K_left = np.array([[left_data['intrinsic']['fx'], 0, left_data['intrinsic']['cx']], [0, left_data['intrinsic']['fy'], left_data['intrinsic']['cy']], [0, 0, 1]])
    D_left = np.array([left_data['intrinsic'].get(k, 0) for k in ['k1', 'k2', 'p1', 'p2', 'k3']])
    K_right = np.array([[right_data['intrinsic']['fx'], 0, right_data['intrinsic']['cx']], [0, right_data['intrinsic']['fy'], right_data['intrinsic']['cy']], [0, 0, 1]])
    D_right = np.array([right_data['intrinsic'].get(k, 0) for k in ['k1', 'k2', 'p1', 'p2', 'k3']])
    
    try:
        with open(os.path.join(base_path, 'depth_scale.txt'), 'r') as f:
            depth_scale = float(f.read())
    except FileNotFoundError:
        depth_scale = 0.001

    for i, timestamp_key in enumerate(common_images):
        print(f"\nProcessing image {i+1}/{len(common_images)}: {timestamp_key}")
        
        rgb_filename = f"realsense_rgb_image_{timestamp_key}.png"
        left_filename = f"thermal_inferno_image_{timestamp_key}.png"
        right_filename = f"thermal_grayscale_image_{timestamp_key}.png"
        depth_filename = f"realsense_depth_raw_{timestamp_key}.png"
        
        rgb_path = os.path.join(base_path, 'realsense', 'rgb', rgb_filename)
        left_path = os.path.join(base_path, 'left_thermal', left_filename)
        right_path = os.path.join(base_path, 'right_thermal', right_filename)
        depth_path = os.path.join(base_path, 'realsense', 'depth_raw', depth_filename)

        rgb_img = cv2.imread(rgb_path)
        left_thermal_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_thermal_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if any(img is None for img in [rgb_img, left_thermal_img, right_thermal_img, depth_img]):
            print(f"Warning: Could not load one or more images for timestamp {timestamp_key}. Skipping.")
            continue

        # --- NEW WORKFLOW ---
        # 1. Undistort the RGB and Depth images first to create a clean canvas.
        undistorted_rgb = cv2.undistort(rgb_img, K_rgb, D_rgb)
        undistorted_depth = cv2.undistort(depth_img, K_rgb, D_rgb)
        
        # 2. Warp thermal images directly to the clean, undistorted RGB view.
        #    We use the undistorted depth map for alignment.
        #    We pass zero distortion for the RGB camera because the target view is now undistorted.
        final_aligned_left = warp_thermal_to_rgb(left_thermal_img, undistorted_depth, K_rgb, np.zeros(5), K_left, D_left, R_left_to_rgb, t_left_to_rgb, depth_scale)
        final_aligned_right = warp_thermal_to_rgb(right_thermal_img, undistorted_depth, K_rgb, np.zeros(5), K_right, D_right, R_right_to_rgb, t_right_to_rgb, depth_scale)

        # 3. Save all the final, clean output files.
        cv2.imwrite(os.path.join(output_dir, 'undistorted_rgb', f'undistorted_rgb_{timestamp_key}.png'), undistorted_rgb)
        cv2.imwrite(os.path.join(output_dir, 'final_aligned_left_thermal', f'aligned_left_{timestamp_key}.png'), final_aligned_left)
        cv2.imwrite(os.path.join(output_dir, 'final_aligned_right_thermal', f'aligned_right_{timestamp_key}.png'), final_aligned_right)
        
        # Also save the undistorted depth for reference
        cv2.imwrite(os.path.join(output_dir, 'undistorted_depth_raw', f'undistorted_depth_raw_{timestamp_key}.png'), undistorted_depth)
        
        # Create and save the overlay for visual verification
        inferno_left = cv2.applyColorMap(final_aligned_left, cv2.COLORMAP_INFERNO)
        inferno_right = cv2.applyColorMap(final_aligned_right, cv2.COLORMAP_INFERNO)
        
        mask_left = final_aligned_left > 0
        mask_right = final_aligned_right > 0
        
        overlay_left = undistorted_rgb.copy()
        overlay_right = undistorted_rgb.copy()

        overlay_left[mask_left] = cv2.addWeighted(undistorted_rgb[mask_left], 0.5, inferno_left[mask_left], 0.5, 0)
        overlay_right[mask_right] = cv2.addWeighted(undistorted_rgb[mask_right], 0.5, inferno_right[mask_right], 0.5, 0)
        
        side_by_side_view = np.hstack((overlay_left, overlay_right))
        cv2.imwrite(os.path.join(output_dir, f'Overlay_Comparison_{timestamp_key}.png'), side_by_side_view)
        print(f"Saved comparison overlay image to {os.path.join(output_dir, f'Overlay_Comparison_{timestamp_key}.png')}")
        
    print("\nVisualization complete.")


def find_validation_image_sets(base_path):
    """
    Scans image directories to find timestamps common to all necessary image types.
    """
    print("\n--- Scanning for Validation Image Sets ---")
    timestamp_pattern = re.compile(r"(\d{8}_\d{6}_\d{9})")
    
    def get_timestamps_from_dir(dir_path):
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            return set()
        timestamps = set()
        for filename in os.listdir(dir_path):
            match = timestamp_pattern.search(filename)
            if match:
                timestamps.add(match.group(1))
        return timestamps

    rgb_timestamps = get_timestamps_from_dir(os.path.join(base_path, 'realsense', 'rgb'))
    left_timestamps = get_timestamps_from_dir(os.path.join(base_path, 'left_thermal'))
    right_timestamps = get_timestamps_from_dir(os.path.join(base_path, 'right_thermal'))
    depth_timestamps = get_timestamps_from_dir(os.path.join(base_path, 'realsense', 'depth_raw'))

    # Simplified check as colorized depth is not directly used in this workflow
    if not all([rgb_timestamps, left_timestamps, right_timestamps, depth_timestamps]):
        print("Error: One or more required image directories (rgb, left_thermal, right_thermal, depth_raw) are empty or missing.")
        return []

    common_validation_stamps = sorted(list(
        rgb_timestamps.intersection(left_timestamps)
                      .intersection(right_timestamps)
                      .intersection(depth_timestamps)
    ))

    if not common_validation_stamps:
        print("Could not find any common timestamps across all required validation image folders.")
    else:
        print(f"Found {len(common_validation_stamps)} complete validation image sets.")
        
    return common_validation_stamps

def main():
    # --- USER CONFIGURATION ---
    use_manual_intrinsics = False
    params_path = '/home/cortex/IRIS/Alignment/yaml_v4_100725'
    rgb_yaml_file = 'RGB_realsense_parameters.yaml'
    left_yaml_file = 'LeftCam_parameters.yaml'
    right_yaml_file = 'RightCam_parameters.yaml'
    
    validation_images_path = "/home/cortex/IRIS/Datasets/Own/RGBDT/v36_Outdoor_Dark_Take1_170725/paired_rgb_left"
    output_dir = "/home/cortex/IRIS/Datasets/Own/RGBDT/v36_Outdoor_Dark_Take1_170725/paired_rgb_left/Alignment Result_v2_RotationTranslation_240725"
    
    manual_rgb_intrinsics = {
        'fx': 386.727, 'fy': 386.165, 'cx': 321.544, 'cy': 243.315,
        'k1': -0.0550349, 'k2': 0.0627269, 'p1': -0.000820398, 'p2': 0.000379441, 'k3': -0.0196893
    }
    manual_left_intrinsics = {
        'fx': 358.009390, 'fy': 356.631007, 'cx': 320.649615, 'cy': 268.477546,
        'k1': -0.210408, 'k2': 0.037092, 'p1': 0.000217, 'p2': 0.000702, 'k3': 0.0
    }
    manual_right_intrinsics = {
        'fx': 358.484823, 'fy': 357.311578, 'cx': 317.492363, 'cy': 267.103537,
        'k1': -0.212959, 'k2': 0.039110, 'p1': 0.000939, 'p2': 0.001243, 'k3': 0.0
    }
    # --- END OF USER CONFIGURATION ---

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'undistorted_rgb'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'final_aligned_left_thermal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'final_aligned_right_thermal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'undistorted_depth_raw'), exist_ok=True)

    print("--- STEP 1: LOADING DATA AND CALCULATING EXTRINSICS ---")
    rgb_data = parse_yaml_file(os.path.join(params_path, rgb_yaml_file))
    left_data = parse_yaml_file(os.path.join(params_path, left_yaml_file))
    right_data = parse_yaml_file(os.path.join(params_path, right_yaml_file))

    if not all([rgb_data, left_data, right_data]):
        print("Aborting due to YAML file loading errors.")
        return

    if use_manual_intrinsics:
        print("\n--- Using MANUALLY DEFINED intrinsic parameters. ---")
        rgb_data['intrinsic'] = manual_rgb_intrinsics
        left_data['intrinsic'] = manual_left_intrinsics
        right_data['intrinsic'] = manual_right_intrinsics
    else:
        print("\n--- Using intrinsic parameters from YAML files. ---")

    R_left_to_rgb, t_left_to_rgb, _ = calculate_and_average_extrinsics(rgb_data, left_data, "RGB", "LeftThermal")
    if R_left_to_rgb is None: return
    save_extrinsics_to_yaml(output_dir, 'extrinsics_left_to_rgb.yml', R_left_to_rgb, t_left_to_rgb, 'LeftThermal', 'RGB')

    R_right_to_rgb, t_right_to_rgb, _ = calculate_and_average_extrinsics(rgb_data, right_data, "RGB", "RightThermal")
    if R_right_to_rgb is None: return
    save_extrinsics_to_yaml(output_dir, 'extrinsics_right_to_rgb.yml', R_right_to_rgb, t_right_to_rgb, 'RightThermal', 'RGB')

    R_left_to_right, t_left_to_right, _ = calculate_and_average_extrinsics(right_data, left_data, "RightThermal", "LeftThermal")
    if R_left_to_right is not None:
        save_extrinsics_to_yaml(output_dir, 'extrinsics_left_to_right.yml', R_left_to_right, t_left_to_right, 'LeftThermal', 'RightThermal')
    else:
        print("Warning: Could not calculate Left-to-Right thermal extrinsics.")

    print("\n--- STEP 2: FINDING VALIDATION IMAGES FROM DISK ---")
    validation_timestamps = find_validation_image_sets(validation_images_path)

    if validation_timestamps:
        print("\n--- STEP 3: APPLYING TRANSFORMATION TO VALIDATION IMAGES ---")
        visualize_and_save_all(
            validation_images_path, 
            output_dir, 
            R_left_to_rgb, t_left_to_rgb, 
            R_right_to_rgb, t_right_to_rgb, 
            rgb_data, left_data, right_data, 
            validation_timestamps
        )
    else:
        print("\nNo validation images found. Skipping visualization.")

if __name__ == '__main__':
    main()