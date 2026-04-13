import os
import glob
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

# --- CONFIGURATION ---
# Point this to the 'best.pt' file from your successful training run
MODEL_PATH = "/opt/homebrew/runs/detect/MVP_runs/logo_detector2/weights/best.pt"
INPUT_DIR = "MVP/dataset/train"
OUTPUT_DIR = "MVP/cleaned_results"

# 1. Initialize Models (We do this once outside the loop to save memory)
print("🔋 Loading Stage 1 (Detector) and Stage 2 (Inpainter)...")
detector = YOLO(MODEL_PATH)
lama = SimpleLama()

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_all():
    # Get all .jpg images from the dataset folder
    image_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    total = len(image_files)
    
    if total == 0:
        print(f"❌ No images found in {INPUT_DIR}. Please check your paths!")
        return

    print(f"📂 Found {total} images. Starting batch cleanup...\n")

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, filename)

        # --- STAGE 1: DETECTION ---
        # verbose=False keeps the console clean
        results = detector(img_path, verbose=False)
        img_orig = Image.open(img_path).convert("RGB")
        w, h = img_orig.size
        
        # Create a black mask (same size as image)
        mask = np.zeros((h, w), dtype=np.uint8)
        detected = False
        
        for r in results:
            for box in r.boxes.xyxy:
                detected = True
                x1, y1, x2, y2 = map(int, box)
                
                # ADD PADDING: We expand the box by 10 pixels 
                # to make sure the inpainter "sees" the background around the logo.
                pad = 10
                cv2.rectangle(mask, 
                              (max(0, x1-pad), max(0, y1-pad)), 
                              (min(w, x2+pad), min(h, y2+pad)), 
                              255, -1)

        # --- STAGE 2: INPAINTING (Only if a logo was found) ---
        if detected:
            mask_pil = Image.fromarray(mask)
            cleaned_img = lama(img_orig, mask_pil)
            cleaned_img.save(save_path)
            status = "✨ Cleaned"
        else:
            # If no logo is found, we just copy the original to the results folder
            img_orig.save(save_path)
            status = "⏩ Skipped"

        print(f"[{i+1}/{total}] {status}: {filename}")

    print(f"\n✅ All 100 images processed! Results are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all()