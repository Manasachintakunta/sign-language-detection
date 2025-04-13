import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import create_model, create_lightweight_model
import json
import time

def train_model(use_lightweight=False):
    # Load data
    X = []
    y = []
    
    # Get all sign directories
    sign_dirs = sorted([d for d in os.listdir("data/processed") if os.path.isdir(os.path.join("data/processed", d))])
    
    if not sign_dirs:
        print("No processed data found. Run preprocessing.py first.")
        return
    
    # Create a mapping from sign to index
    sign_to_idx = {sign: idx for idx, sign in enumerate(sign_dirs)}
    idx_to_sign = {idx: sign for sign, idx in sign_to_idx.items()}
    
    print(f"Found {len(sign_dirs)} sign classes: {', '.join(sign_dirs)}")
    
    # Load images and labels
    for sign in sign_dirs:
        sign_idx = sign_to_idx[sign]
        
        image_files = [f for f in os.listdir(f"data/processed/{sign}") if f.endswith('.jpg')]
        
        if not image_files:
            print(f"Warning: No images found for sign {sign}")
            continue
            
        print(f"Loading {len(image_files)} images for sign {sign}")
        
        for img_file in image_files:
            img_path = f"data/processed/{sign}/{img_file}"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            X.append(img)
            y.append(sign_idx)
    
    if not X:
        print("No valid images found. Check your processed data.")
        return
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Data loaded: {X.shape[0]} images, {len(set(y))} classes")
    
    # Normalize images
    X = X / 255.0
    
    # Reshape for CNN input
    X = X.reshape(X.shape[0], 128, 128, 1)
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=len(sign_dirs))
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    
    # Create model
    if use_lightweight:
        model = create_lightweight_model(len(sign_dirs))
        model_name = "sign_language_model_lightweight.h5"
    else:
        model = create_model(len(sign_dirs))
        model_name = "sign_language_model.h5"
    
    # Create model directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        f"models/{model_name}",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the sign mappings
    with open("models/sign_mapping.json", "w") as f:
        json.dump({
            "sign_to_idx": sign_to_idx,
            "idx_to_sign": idx_to_sign
        }, f)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig("models/training_history.png")
    plt.show()
    
    print("Model training complete!")
    print(f"Model saved as models/{model_name}")
    print(f"Sign mapping saved as models/sign_mapping.json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train sign language detection model')
    parser.add_argument('--lightweight', action='store_true', help='Use lightweight model for systems with limited resources')
    
    args = parser.parse_args()
    
    train_model(use_lightweight=args.lightweight)