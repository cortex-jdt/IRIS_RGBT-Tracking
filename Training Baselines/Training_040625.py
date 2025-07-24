from ultralytics import YOLO
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.rand(2, 3).cuda()) # Optional: Test if a tensor can be moved to GPU

if torch.cuda.is_available():
    print("NVIDIA GPU is available for PyTorch!")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}") # Prints the name of the first GPU
else:
    print("NVIDIA GPU is NOT available for PyTorch. Training will likely run on CPU.")

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Explicitly set device to GPU if available
if torch.cuda.is_available():
    device = '1' # Use the first GPU (index 0) or second GPU (index 1)
    print("Attempting to train on GPU.")
else:
    device = 'cpu'
    print("No GPU available, training on CPU.")

# Train the model
dataset = "/media/act/Javier/Datasets/New_170725/LLVIP_RGB/data.yaml"
results = model.train(data=dataset, 
                      epochs = 300,           # Increased epochs (let patience handle stopping)
                      patience = 50,          # Stop if no improvement for 50 epochs
                      batch = 64,
                      weight_decay = 0.001,            
                      val = True, 
                      plots = True,
                      seed = 13321048, 
                      device=device)

# Run with 1st GPU (CUDA_VISIBLE_DEVICES=0) or 2nd GPU (CUDA_VISIBLE_DEVICES=1)
# Example:  CUDA_VISIBLE_DEVICES=1 python3 Training_040625.py 