# IRIS Research Internship Project (by Javier Dustin Tambunan)
## General Overview
This readme file explains everything regarding programs being used from capturing dataset to YOLO11n model training & tracking test. This readme file is created on Wednesday, July 23rd 2025.

## A. Capturing Dataset
### Overview
Program below will capture in png files continuously (with "A_capture_RGBDT_cont.cpp") OR intermittent (with "A_capture_RGBDT.cpp").

### Steps
1. Git clone "iray_capturer" repo and go to v4l2 folder.
```
git clone https://github.com/JzHuai0108/iray_capturer.git
cd v4l2
```
2. a. If you want to capture images continuously, build and run A_capture_RGBDT_cont.cpp by following the instructions commented at the top section of the program. More instructions are provided at the top section too.

In this program, the Realsense RGBD is set to 640x480px 60fps and aligned already. You can re-configure the setting in "void producer_realsense(const std::string& rs_serial, int id)" function.

Note: this program is recommended for capturing training datasets.

```
 * To build:
 * sudo apt-get install libv4l-dev libopencv-dev librealsense2-dev
 * g++ -std=c++17 -o A_capture_RGBDT_cont A_capture_RGBDT_cont.cpp -lv4l2 $(pkg-config --cflags --libs opencv4) -lrealsense2 -pthread
 * To run:
 * LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 ./A_capture_RGBDT_cont [left_cam_idx] [right_cam_idx] [thermal_fps] [output_dir] [realsense_serial_optional]
 * Example for 25 FPS Thermal:
 * LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 ./A_capture_RGBDT_cont 8 10 25 /home/cortex/IRIS/Datasets/Drone_Capture_01
 */
```

2. b. If you want to capture images by pressing key, build and run A_capture_RGBDT.cpp by following the instructions commented at the top section of the program. More instructions are provided at the top section too.

You can re-configure the setting in "void producer_realsense(const std::string& rs_serial, int id)" function.

Note: this program is recommended for capturing calibration datasets.

```
 * To build:
 * sudo apt-get install libv4l-dev libopencv-dev librealsense2-dev
 * g++ -std=c++17 -o A_capture_RGBDT A_capture_RGBDT.cpp -lv4l2 $(pkg-config --cflags --libs opencv4) -lrealsense2 -pthread
 * * To run (you may need LD_PRELOAD on some systems):
 * LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 ./A_capture_RGBDT [right_cam_idx] [left_cam_idx] [sync_tolerance_us] [fps] [output_dir]
 * Example:
 * LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 ./A_capture_RGBDT 8 10 17000 30 /home/cortex/IRIS/Datasets/Own/RGBDT/v1_Trial_010725
 */
```

## B. Dataset Pairing
### Overview
The RGBD and thermal images are not paired yet. Thus, we'll need a program to do that.

### Steps
1. Go to the "Pairing_Images_v4_090725.py" program path
```
cd Pairing
```

2. Modify user configuration section based on your needs. Then, run the program.
```
    # --- USER CONFIGURATION ---
    dataset_base_path = Path("/home/cortex/IRIS/Datasets/Own/RGBDT/v37_Outdoor_Dark_Take2_170725")
    log_file_name = "capture_log_20250717_195730.csv"
    max_allowed_delay_ms = 2 # Set the maximum sync delay in milliseconds

    output_folder_left = "paired_rgb_left"
    output_csv_left = "pairs_rgb_left.csv"
    
    output_folder_right = "paired_rgb_right"
    output_csv_right = "pairs_rgb_right.csv"
    # --- END OF CONFIGURATION ---
```

3. The program outputs two folders: "paired_rgb_left" and "paired_rgb_right". The "paired_rgb_left" folder contains RGB images paired with the nearest-in-time images from the left thermal camera, within a "max_allowed_delay_ms". However, it also provides right thermal images. Similarly, the "paired_rgb_right" folder contains RGB images paired with the nearest images from the right thermal camera.

## C. Camera Calibration
### Overview
Before you perform the RGB-T alignment, we need intrinsic & extrinsic parameters of all cameras. MATLAB program is created to achieve this purpose since it's already well-known and easy-to-use. You can create Python or C++ program yourself based on this MATLAB program, but make sure that the yaml outputs have the same content since we'll use these yaml files for alignment later.

### Steps
1. Go to "LeftCam_params.m", "RGB_params.m", and "RightCam_params.m" paths
```
cd Calibration
```

2. Run each program according to your needs. You also can use MATLAB Camera Calibration App to have a glimpse of how the calibration works. I also hope the MATLAB codes are self-explanatory.


## D. Alignment
### Overview
There are 3 programs provided: 
(1) Rotation + Translation with Inpaint: recommended except in outdoor, dark scene. Program: "B_align_images_v9_AlignedUndistortedRGBDT_130725.py"
(2) Rotation + Translation: recommended for general use. Program: "B_align_images_v10_NoInpaint_RotationandTranslation_140725.py"
(3) Rotation: recommended if all captured objects are far away from the cameras. Program: "B_align_images_v11_NoInpaint_RotationOnly_130725.py"

### Steps
1. Go to the program path based on your preference.
```
cd Alignment
```

2. Modify the user configuration section based on your needs. For example, in "B_align_images_v9_AlignedUndistortedRGBDT_130725.py":
```
    # --- USER CONFIGURATION ---
    use_manual_intrinsics = False
    params_path = '/home/cortex/IRIS/Alignment/yaml_v4_100725'
    rgb_yaml_file = 'RGB_realsense_parameters.yaml'
    left_yaml_file = 'LeftCam_parameters.yaml'
    right_yaml_file = 'RightCam_parameters.yaml'
    validation_images_path = "/home/cortex/IRIS/Datasets/Own/RGBDT/v36_Outdoor_Dark_Take1_170725/paired_rgb_left"
    output_dir = "/home/cortex/IRIS/Datasets/Own/RGBDT/v36_Outdoor_Dark_Take1_170725/paired_rgb_left/Alignment_Result_v1_240725"
    
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
```
As you can see, you also can configure the intrinsics parameters manually if needed.

3. Run the program.

## E. Upload to Roboflow
### Overview
To upload the dataset from local to Roboflow in bulk size, you can use "UploadToRoboflow.py" program. In this program, you upload the RGB and thermal images separately in different Roboflow projects.

### Steps
1. Go to program path
```
cd ToRoboflow
```
2. Run the program after you modify the user configuration section.

IMPORTANT Notes: this program initially created to upload LLVIP, FLIR, and Tokyo datasets, and has never been used to upload author's dataset yet. Hence, some modifications may be needed. Make sure the RGB-T paired images have similar filenames to identify them as a pair. For example, an RGB image named "171025_10072025.png" is paired with thermal image named "171025_10072025.png" too.


## F. Annotation in Roboflow
### Overview
There is no program provided since you conduct the annotations in Roboflow. To speed up the annotation process, you can train detection model in Roboflow after you annotate around 100-200 images manually. For further instructions, you can access these link: https://docs.roboflow.com/annotate/ai-labeling/automated-annotation-with-autodistill

## G. RGB-T Pairs Filename Matching
### Overview
After you've done with annotation then downloading the dataset, do note that Roboflow will give unique names for each images. Because of this, when you want to use the dataset, remove this unique name or make sure your training/testing/tracking/etc. program call the basename only.

For example, if the RGB & thermal images have "FLIR_10227_RGB_jpg.rf.b1b1afe760674d99f39c441c617e1b98.jpg" and "FLIR_10227_PreviewData_jpeg.rf.1cef4ef256f52a120b3149e9327fd0d0.jpg" respectively, then make sure you only use the same part of the their name, e.g. "FLIR_10227".

## H. YOLO11n Detection Model Baseline Training
### Overview
Train a YOLO11n baseline model is quite simple. "Training_040625.py" program gives you the example how to train ~~your dragon~~ detection model baseline with/without dedicated GPU.

### Steps
1. Go to program path
```
cd Training Baselines
```
2. Modify the dataset path, pre-trained weight path, and training configuration based on your needs. For further information regarding the training settings can be accessed from this link: https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings

3. Run the program. Then, you can access the weight file & training results too.

## I. Tracking Test
### Overview
To conduct tracking test based on your trained detection model, you can use "Tracking_v2_ImageFormat_210725.py". YOLO11n tracking docs can be found here: https://docs.ultralytics.com/modes/track/#available-trackers

### Steps
1. Go to program path
```
cd Tracking
```

2. Change these configuration lines based on your needs:
```
model = YOLO("/home/javier/Javier/runs/detect/train7_LLVIP_RGB_v2_190725/weights/best.pt")
input_folder = "/media/act/Javier/Datasets/New_170725/RGB-T234_Partial/nightthreepeople/visible" # <--- IMPORTANT: Path to your image folder
desired_fps = 30 # <--- Set the frame rate for the output video
output_directory = "/media/act/Javier/Tracking_result" # <--- CHANGE THIS PATH
results = model.track(frame, persist=True, conf=0.5, iou=0.5, tracker="/home/javier/Javier/botsort_ReID.yaml")
```

## J. Late Fusion Model (So Far)
### Overview
"Fusion_Late_v2_150625.py" program is one of the example of late fusion model. However, this program is the first iteration of author's attempt to learn fusion detection model. Hence, it needs to be validated further.

### Steps
1. Go to program path
```
cd Training Fusion
```
2. Modify the user configuration section based on your needs.
```
# Load models
model_rgb = YOLO('/home/javier/Javier/runs/detect/train_yolo11n_LLVIP-FLIR-Tokyo_Visible_1_080625/weights/best.pt')
model_thermal = YOLO('/home/javier/Javier/runs/detect/train_yolo11n_LLVIP-FLIR-Tokyo_Infrared_1_070625/weights/best.pt')

# Define folders
rgb_folder = '/media/act/Javier_datasets/CVC14/CVC-14/Night/Visible/NewTest/FramesPos'
thermal_folder = '/media/act/Javier_datasets/CVC14/CVC-14/Night/FIR/NewTest/FramesPos'
output_folder = '/home/javier/Javier/Fusion_Late_v2_Result'
fusion_label_folder = '/home/javier/Javier/Fusion_Late_v2_Labels'

# Create output folders if not exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(fusion_label_folder, exist_ok=True)

# Process only .tif files
rgb_filenames = sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith('.tif')])
```

3. Run the program to start the training.


IMPORTANT Notes: all dependencies used are not listed here yet. Currently, you can install dependencies based on the error given by the programs when you run them. Sorry for the inconvenience caused.

## Dataset
These are the datasets that author used:
(1) Dataset collected by author: https://drive.google.com/drive/folders/1XnWUcJ7O3L-jIDybb8T8xaLdXuuaJKQs?usp=sharing
(2) Modified LLVIP, FLIR, and Tokyo Dataset: https://drive.google.com/drive/folders/1ru-CN9KJ0R7hk3yb4M4_ndNthT64QP0Y?usp=sharing

The main structure of Dataset (1):
> "left_thermal" folder: thermal images from left thermal camera in author's camera setup
> "realsense" folder: RGBD images from Intel Realsense D455f in author's camera setup
> "right_thermal" folder: thermal images from right thermal camera in author's camera setup
> "capture_log_{date}_{time}.csv" file: data log

There are other folders too, generated from section "B. Dataset Pairing", "D. Alignment", etc.

The structure of Dataset (2) is the same as common Roboflow dataset.