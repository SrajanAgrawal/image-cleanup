import os
import random
import glob
from PIL import Image, ImageEnhance

def generate_100_from_samples(raw_dir, logos_dir, output_dir, target_count=100):
    # This creates the folders if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get lists of images
    bg_paths = glob.glob(os.path.join(raw_dir, "*.jpg"))
    logo_paths = glob.glob(os.path.join(logos_dir, "*.png"))
    
    # Debugging prints
    print(f"Found {len(bg_paths)} background images in: {raw_dir}")
    print(f"Found {len(logo_paths)} logo images in: {logos_dir}")

    if len(bg_paths) == 0 or len(logo_paths) == 0:
        print("❌ Error: Could not find images. Check your folder paths!")
        return

    count = 0
    while count < target_count:
        # Pick random samples
        bg = Image.open(random.choice(bg_paths)).convert("RGB")
        logo = Image.open(random.choice(logo_paths)).convert("RGBA")
        
        # --- 1. Augment Background ---
        if random.random() > 0.5:
            bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(random.uniform(0.8, 1.2))

        # --- 2. Randomize Logo ---
        bg_w, bg_h = bg.size
        scale = random.uniform(0.10, 0.25) # Made logos slightly larger for MVP visibility
        logo_w = int(bg_w * scale)
        logo_h = int(logo_w * (logo.size[1] / logo.size[0]))
        logo = logo.resize((logo_w, logo_h))
        
        x = random.randint(10, max(11, bg_w - logo_w - 10))
        y = random.randint(10, max(11, bg_h - logo_h - 10))
        
        # --- 3. Save Output & Label ---
        bg.paste(logo, (x, y), logo)
        
        name = f"synthetic_{count:03d}"
        bg.save(os.path.join(output_dir, f"{name}.jpg"))
        
        # Create YOLO label
        with open(os.path.join(output_dir, f"{name}.txt"), 'w') as f:
            cx = (x + logo_w/2) / bg_w
            cy = (y + logo_h/2) / bg_h
            f.write(f"0 {cx} {cy} {logo_w/bg_w} {logo_h/bg_h}\n")
            
        count += 1
    print(f"✅ Success! Generated {target_count} images in {output_dir}")

if __name__ == "__main__":
    # We use absolute-style relative paths to ensure it works from ModelTraining/
    base_path = "MVP"
    raw = os.path.join(base_path, "raw")
    logos = os.path.join(base_path, "logos")
    output = os.path.join(base_path, "dataset", "train")
    
    generate_100_from_samples(raw, logos, output)