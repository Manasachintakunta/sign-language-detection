import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_images():
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
    
    # Get all sign directories
    sign_dirs = [d for d in os.listdir("data/raw") if os.path.isdir(os.path.join("data/raw", d))]
    
    for sign in sign_dirs:
        print(f"Processing images for sign: {sign}")
        
        # Create processed directory for this sign
        processed_dir = f"data/processed/{sign}"
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        # Get all images for this sign
        image_files = [f for f in os.listdir(f"data/raw/{sign}") if f.endswith('.jpg')]
        
        for img_file in tqdm(image_files):
            img_path = f"data/raw/{sign}/{img_file}"
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding to better isolate hand
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Resize to consistent dimensions
            resized = cv2.resize(thresh, (128, 128))
            
            # Save processed image
            cv2.imwrite(f"{processed_dir}/{img_file}", resized)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_images()
# Enhanced preprocessing pipeline - updated April 2025
