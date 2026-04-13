from ultralytics import YOLO
import os

def start_training():
    # 1. Load a pre-trained Nano model
    # yolov10n is the latest and very fast on Mac
    model = YOLO("yolov10n.pt") 

    # 2. Train the model
    # We use 25 epochs - enough for 100 images to show results
    # device='mps' uses your MacBook's GPU
    model.train(
        data="MVP/data.yaml", 
        epochs=25, 
        imgsz=640, 
        device="mps",
        project="MVP_runs",  # Saves results in an MVP_runs folder
        name="logo_detector"
    )
    
    print("\n✅ Training Complete!")
    print("Your 'brain' file is located at: MVP_runs/logo_detector/weights/best.pt")

if __name__ == "__main__":
    start_training()