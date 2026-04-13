import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

# 1. Load the Stage 1 Brain (Detector)
# Note: Use the path from your successful output
DETECTOR_PATH = "/opt/homebrew/runs/detect/MVP_runs/logo_detector2/weights/best.pt"
detector = YOLO(DETECTOR_PATH)

# 2. Load the Stage 2 Artist (Inpainter)
lama = SimpleLama()

def clean_image(image_path, output_path):
    # --- STAGE 1: DETECTION ---
    results = detector(image_path)
    img_orig = Image.open(image_path).convert("RGB")
    w, h = img_orig.size
    
    # Create a blank black mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    detected = False
    for r in results:
        for box in r.boxes.xyxy:
            detected = True
            # Get coordinates and expand them slightly (5px) for a cleaner edge
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1-5, y1-5), (x2+5, y2+5), 255, -1)
    
    if not detected:
        print("No logo detected. Saving original.")
        img_orig.save(output_path)
        return

    # --- STAGE 2: INPAINTING ---
    print("Logo found! Healing the image...")
    mask_pil = Image.fromarray(mask)
    result = lama(img_orig, mask_pil)
    
    # Save the final result
    result.save(output_path)
    print(f"✨ Cleaned image saved to: {output_path}")

if __name__ == "__main__":
    # Test it on one of your synthetic images!
    test_img = "MVP/dataset/train/synthetic_000.jpg"
    clean_image(test_img, "final_mvp_test.png")