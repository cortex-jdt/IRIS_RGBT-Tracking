import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import glob # Import glob to easily find image files

# Load the YOLO model
model = YOLO("/home/javier/Javier/runs/detect/train7_LLVIP_RGB_v2_190725/weights/best.pt")
# Training weights
# (Recommended) YOLO11n Infrared (LLVIP-FLIR-Tokyo): /home/javier/Javier/runs/detect/train_yolo11n_LLVIP-FLIR-Tokyo_Infrared_1_070625/weights/best.pt
# (Recommended) YOLO11n Visible (LLVIP-FLIR-Tokyo): /home/javier/Javier/runs/detect/train_yolo11n_LLVIP-FLIR-Tokyo_Visible_1_080625/weights/best.pt
# YOLO11n Infrared (LLVIP): /home/javier/Javier/runs/detect/train7_LLVIP_Infrared_v2_190725/weights/best.pt
# YOLO11n Visible (LLVIP): /home/javier/Javier/runs/detect/train7_LLVIP_RGB_v2_190725/weights/best.pt

# --- MODIFIED: INPUT SETUP FOR IMAGE SEQUENCE ---
# >>> NEW: DEFINE YOUR INPUT FOLDER AND DESIRED OUTPUT FPS HERE <<<
input_folder = "/media/act/Javier/Datasets/New_170725/RGB-T234_Partial/nightthreepeople/visible" # <--- IMPORTANT: Path to your image folder
desired_fps = 30 # <--- Set the frame rate for the output video

# Get a sorted list of all image files in the input folder
# This supports common image formats like .png, .jpg, .jpeg, .bmp, .tif, .tiff
image_files = sorted(glob.glob(os.path.join(input_folder, '*.[pP][nN][gG]')) + 
                     glob.glob(os.path.join(input_folder, '*.[jJ][pP][gG]')) + 
                     glob.glob(os.path.join(input_folder, '*.[jJ][pP][eE][gG]')) +
                     glob.glob(os.path.join(input_folder, '*.[bB][mM][pP]')) +
                     glob.glob(os.path.join(input_folder, '*.[tT][iI][fF]')) +
                     glob.glob(os.path.join(input_folder, '*.[tT][iI][fF][fF]')))

# Check if any images were found
if not image_files:
    print(f"Error: No image files found in the specified folder: {input_folder}")
    exit()

print(f"Found {len(image_files)} images to process.")

# --- MODIFIED: VIDEO WRITER SETUP ---
# Read the first image to get its dimensions (width, height)
first_frame = cv2.imread(image_files[0])
if first_frame is None:
    print(f"Error: Could not read the first image: {image_files[0]}")
    exit()
frame_height, frame_width, _ = first_frame.shape

# >>> NEW: DEFINE YOUR DESIRED OUTPUT DIRECTORY HERE <<<
output_directory = "/media/act/Javier/Tracking_result" # <--- CHANGE THIS PATH
os.makedirs(output_directory, exist_ok=True) # Create the directory if it doesn't exist

# Generate a unique base filename with a readable timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_filename = f"tracking_result{timestamp}.avi"
output_path = os.path.join(output_directory, base_filename)

# Define the output video codec and initialize the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, desired_fps, (frame_width, frame_height))
print(f"Saving tracking video to: {os.path.abspath(output_path)}")
# --- END OF MODIFIED SETUP ---

# Loop through the image files
for image_path in image_files:
    # Read the current image file
    frame = cv2.imread(image_path)

    # Check if the frame was read successfully
    if frame is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, conf=0.5, iou=0.5, tracker="/home/javier/Javier/botsort_ReID.yaml")

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)

    # # Display the annotated frame
    # cv2.imshow("YOLO Tracking", annotated_frame)

    # # Break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     print("Processing interrupted by user.")
    #     break

# --- MODIFIED: RELEASE EVERYTHING ---
print("Processing finished.")
out.release() # IMPORTANT: Release the writer to finalize and save the video
cv2.destroyAllWindows()
print(f"Video saved successfully to {os.path.abspath(output_path)}")