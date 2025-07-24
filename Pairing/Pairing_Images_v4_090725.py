# pre_process_and_pair.py
#
# WHAT IT DOES: (MODIFIED FOR INDEPENDENT PAIRING WITH FULL INFO)
# 1. Performs two separate searches to create two independent datasets.
# 2. Dataset 1 (RGB-Left): Finds all RGB and Left Thermal pairs with a <2ms sync delay.
#    For each valid pair, it also finds the nearest Right Thermal frame.
# 3. Dataset 2 (RGB-Right): Finds all RGB and Right Thermal pairs with a <2ms sync delay.
#    For each valid pair, it also finds the nearest Left Thermal frame.
# 4. Applies strict uniqueness filters to each dataset independently to ensure a 1-to-1-to-1 mapping.
# 5. Saves each dataset into its own folder, each with a 'matched_pairs.csv' file
#    containing all relevant time delay information.

import pandas as pd
import numpy as np
import os
import datetime
import shutil
from pathlib import Path

def parse_host_timestamp(ts_str):
    """Converts the 'seconds_nanoseconds' string to a numeric float timestamp."""
    try:
        parts = str(ts_str).split('_')
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 1e9
    except (ValueError, IndexError):
        return 0.0
    return 0.0

def to_filename_timestamp(numeric_ts):
    """
    Converts a numeric float timestamp to a filename-safe string WITH 9-digit nanoseconds.
    """
    if numeric_ts == 0.0:
        return "invalid_timestamp"
    
    seconds = int(numeric_ts)
    nanoseconds = int((numeric_ts - seconds) * 1_000_000_000)
    
    dt_object = datetime.datetime.fromtimestamp(seconds)
    date_time_part = dt_object.strftime('%Y%m%d_%H%M%S')
    
    return f"{date_time_part}_{nanoseconds:09d}"

def create_paired_directories(base_path):
    """Creates the necessary directory structure for the paired files."""
    print(f"Creating directory structure in: {base_path}")
    try:
        os.makedirs(base_path / "left_thermal" / "temp_raw", exist_ok=True)
        os.makedirs(base_path / "right_thermal" / "temp_raw", exist_ok=True)
        os.makedirs(base_path / "realsense" / "rgb", exist_ok=True)
        os.makedirs(base_path / "realsense" / "depth_raw", exist_ok=True)
        os.makedirs(base_path / "realsense" / "depth_colorized", exist_ok=True)
        print("Directory structure created successfully.")
    except OSError as e:
        print(f"FATAL ERROR: Could not create directories: {e}")
        raise

def main():
    # --- USER CONFIGURATION ---
    dataset_base_path = Path("/home/cortex/IRIS/Datasets/Own/RGBDT/v37_Outdoor_Dark_Take2_170725")
    log_file_name = "capture_log_20250717_195730.csv"
    max_allowed_delay_ms = 2 # Set the maximum sync delay in milliseconds

    output_folder_left = "paired_rgb_left"
    output_csv_left = "pairs_rgb_left.csv"
    
    output_folder_right = "paired_rgb_right"
    output_csv_right = "pairs_rgb_right.csv"
    # --- END OF CONFIGURATION ---

    log_file_path = dataset_base_path / log_file_name
    
    paired_path_left = dataset_base_path / output_folder_left
    output_csv_path_left = paired_path_left / output_csv_left
    
    paired_path_right = dataset_base_path / output_folder_right
    output_csv_path_right = paired_path_right / output_csv_right

    print(f"--- Starting Pre-Processing ---")
    
    print(f"Reading raw log file: {log_file_path}")
    try:
        df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Log file not found at '{log_file_path}'")
        return

    df['numeric_ts'] = df['precise_host_timestamp'].apply(parse_host_timestamp)

    df_rs = df[df['camera_name'] == 'realsense'].copy().dropna(subset=['numeric_ts'])
    df_left = df[df['camera_name'] == 'left_thermal'].copy().dropna(subset=['numeric_ts'])
    df_right = df[df['camera_name'] == 'right_thermal'].copy().dropna(subset=['numeric_ts'])

    if df_rs.empty or df_left.empty or df_right.empty: return

    # --- PROCESS RGB-LEFT PAIRS ---
    print("\n--- Finding RGB-Left Thermal Pairs ---")
    matched_left_data = []
    for index, rs_row in df_rs.iterrows():
        rs_ts_numeric = rs_row['numeric_ts']
        time_diffs_left = np.abs(df_left['numeric_ts'].values - rs_ts_numeric)
        best_left_idx = np.argmin(time_diffs_left)
        delay_left_ms = time_diffs_left[best_left_idx] * 1000

        if delay_left_ms < max_allowed_delay_ms:
            best_left_row = df_left.iloc[best_left_idx]
            
            # Now find the nearest right_thermal frame to this valid pair's RGB frame
            time_diffs_right = np.abs(df_right['numeric_ts'].values - rs_ts_numeric)
            best_right_idx = np.argmin(time_diffs_right)
            best_right_row = df_right.iloc[best_right_idx]
            
            # Calculate all delays
            delay_right_ms = time_diffs_right[best_right_idx] * 1000
            delay_left_right_ms = (best_left_row['numeric_ts'] - best_right_row['numeric_ts']) * 1000
            
            matched_left_data.append({
                'realsense_frame_orig': str(rs_row['frame_timestamp']),
                'left_thermal_frame_orig': str(best_left_row['frame_timestamp']),
                'right_thermal_frame_orig': str(best_right_row['frame_timestamp']),
                'delay_rgb_vs_left_ms': delay_left_ms,
                'delay_rgb_vs_right_ms': delay_right_ms,
                'delay_left_vs_right_ms': delay_left_right_ms
            })
    
    left_pairs_df = pd.DataFrame(matched_left_data)
    print(f"Found {len(left_pairs_df)} potential RGB-Left pairs with < {max_allowed_delay_ms}ms delay.")
    left_pairs_df.drop_duplicates(subset=['realsense_frame_orig', 'left_thermal_frame_orig', 'right_thermal_frame_orig'], keep='first', inplace=True)
    left_pairs_df = left_pairs_df.sort_values(by='realsense_frame_orig').reset_index(drop=True)
    print(f"Found {len(left_pairs_df)} final unique RGB-Left pairs.")
    
    create_paired_directories(paired_path_left)
    left_pairs_df.to_csv(output_csv_path_left, index=False)
    
    print(f"\nCopying {len(left_pairs_df)} RGB-Left-Right triplets...")
    for index, row in left_pairs_df.iterrows():
        try:
            output_filename_key = to_filename_timestamp(pd.to_numeric(row['realsense_frame_orig'].replace('_', '.')))
            src_rs_ts = row['realsense_frame_orig']
            src_left_ts = row['left_thermal_frame_orig']
            src_right_ts = row['right_thermal_frame_orig']
            
            shutil.copy(dataset_base_path / f"realsense/rgb/realsense_rgb_image_{src_rs_ts}.png", paired_path_left / f"realsense/rgb/realsense_rgb_image_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"realsense/depth_raw/realsense_depth_raw_{src_rs_ts}.png", paired_path_left / f"realsense/depth_raw/realsense_depth_raw_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"realsense/depth_colorized/realsense_depth_colorized_image_{src_rs_ts}.png", paired_path_left / f"realsense/depth_colorized/realsense_depth_colorized_image_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"left_thermal/thermal_inferno_image_{src_left_ts}.png", paired_path_left / f"left_thermal/thermal_inferno_image_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"left_thermal/temp_raw/temp_raw_{src_left_ts}.xml", paired_path_left / f"left_thermal/temp_raw/temp_raw_{output_filename_key}.xml")
            shutil.copy(dataset_base_path / f"right_thermal/thermal_grayscale_image_{src_right_ts}.png", paired_path_left / f"right_thermal/thermal_grayscale_image_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"right_thermal/temp_raw/temp_raw_{src_right_ts}.xml", paired_path_left / f"right_thermal/temp_raw/temp_raw_{output_filename_key}.xml")
        except Exception as e:
            print(f"Error copying left-based pair files: {e}")

    # --- PROCESS RGB-RIGHT PAIRS ---
    print("\n--- Finding RGB-Right Thermal Pairs ---")
    matched_right_data = []
    for index, rs_row in df_rs.iterrows():
        rs_ts_numeric = rs_row['numeric_ts']
        time_diffs_right = np.abs(df_right['numeric_ts'].values - rs_ts_numeric)
        best_right_idx = np.argmin(time_diffs_right)
        delay_right_ms = time_diffs_right[best_right_idx] * 1000

        if delay_right_ms < max_allowed_delay_ms:
            best_right_row = df_right.iloc[best_right_idx]

            # Now find the nearest left_thermal frame to this valid pair's RGB frame
            time_diffs_left = np.abs(df_left['numeric_ts'].values - rs_ts_numeric)
            best_left_idx = np.argmin(time_diffs_left)
            best_left_row = df_left.iloc[best_left_idx]

            # Calculate all delays
            delay_left_ms = time_diffs_left[best_left_idx] * 1000
            delay_left_right_ms = (best_left_row['numeric_ts'] - best_right_row['numeric_ts']) * 1000
            
            matched_right_data.append({
                'realsense_frame_orig': str(rs_row['frame_timestamp']),
                'left_thermal_frame_orig': str(best_left_row['frame_timestamp']),
                'right_thermal_frame_orig': str(best_right_row['frame_timestamp']),
                'delay_rgb_vs_left_ms': delay_left_ms,
                'delay_rgb_vs_right_ms': delay_right_ms,
                'delay_left_vs_right_ms': delay_left_right_ms
            })

    right_pairs_df = pd.DataFrame(matched_right_data)
    print(f"Found {len(right_pairs_df)} potential RGB-Right pairs with < {max_allowed_delay_ms}ms delay.")
    right_pairs_df.drop_duplicates(subset=['realsense_frame_orig', 'left_thermal_frame_orig', 'right_thermal_frame_orig'], keep='first', inplace=True)
    right_pairs_df = right_pairs_df.sort_values(by='realsense_frame_orig').reset_index(drop=True)
    print(f"Found {len(right_pairs_df)} final unique RGB-Right pairs.")

    create_paired_directories(paired_path_right)
    right_pairs_df.to_csv(output_csv_path_right, index=False)

    print(f"\nCopying {len(right_pairs_df)} RGB-Right-Left triplets...")
    for index, row in right_pairs_df.iterrows():
        try:
            output_filename_key = to_filename_timestamp(pd.to_numeric(row['realsense_frame_orig'].replace('_', '.')))
            src_rs_ts = row['realsense_frame_orig']
            src_left_ts = row['left_thermal_frame_orig']
            src_right_ts = row['right_thermal_frame_orig']

            shutil.copy(dataset_base_path / f"realsense/rgb/realsense_rgb_image_{src_rs_ts}.png", paired_path_right / f"realsense/rgb/realsense_rgb_image_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"realsense/depth_raw/realsense_depth_raw_{src_rs_ts}.png", paired_path_right / f"realsense/depth_raw/realsense_depth_raw_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"realsense/depth_colorized/realsense_depth_colorized_image_{src_rs_ts}.png", paired_path_right / f"realsense/depth_colorized/realsense_depth_colorized_image_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"left_thermal/thermal_inferno_image_{src_left_ts}.png", paired_path_right / f"left_thermal/thermal_inferno_image_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"left_thermal/temp_raw/temp_raw_{src_left_ts}.xml", paired_path_right / f"left_thermal/temp_raw/temp_raw_{output_filename_key}.xml")
            shutil.copy(dataset_base_path / f"right_thermal/thermal_grayscale_image_{src_right_ts}.png", paired_path_right / f"right_thermal/thermal_grayscale_image_{output_filename_key}.png")
            shutil.copy(dataset_base_path / f"right_thermal/temp_raw/temp_raw_{src_right_ts}.xml", paired_path_right / f"right_thermal/temp_raw/temp_raw_{output_filename_key}.xml")
        except Exception as e:
            print(f"Error copying right-based pair files: {e}")

    print(f"\n--- Pre-Processing Complete! ✅ ---")
    print(f"Created two separate datasets:")
    print(f"  - RGB-Left Thermal Pairs: {paired_path_left}")
    print(f"  - RGB-Right Thermal Pairs: {paired_path_right}")

if __name__ == '__main__':
    main()