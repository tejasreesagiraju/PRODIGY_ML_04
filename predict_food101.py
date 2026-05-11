"""
Food-101 Prediction Script
Use a trained model to predict food categories in images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Configuration
MODEL_PATH = 'food101_model.h5'
IMAGE_SIZE = (224, 224)

print("=" * 60)
print("FOOD-101 PREDICTION TOOL")
print("=" * 60)

# ============================================================================
# 1. LOAD MODEL
# ============================================================================
print("\n1. Loading trained model...")

if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print(f"   ✓ Model loaded: {MODEL_PATH}")
else:
    print(f"   ✗ Model not found: {MODEL_PATH}")
    print("   Please train the model first using train_food101_model.py")
    exit(1)

# ============================================================================
# 2. LOAD CLASS NAMES
# ============================================================================
print("\n2. Loading class names...")

meta_path = "d:/task4/food-101/meta/classes.txt"
if os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"   ✓ Loaded {len(classes)} classes")
else:
    # Create dummy class names if file not found
    classes = [f"Food_{i}" for i in range(101)]
    print(f"   ⚠ Using default class names")

# ============================================================================
# 3. PREDICTION FUNCTION
# ============================================================================
def predict_image(image_path, top_k=5):
    """
    Predict food category for a given image
    
    Args:
        image_path: Path to image file
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    if not os.path.exists(image_path):
        print(f"   ✗ Image not found: {image_path}")
        return None
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, 0)
    
    # Make prediction
    predictions = model.predict(img_batch, verbose=0)
    
    # Get top-k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    results = {}
    
    for rank, idx in enumerate(top_indices, 1):
        confidence = float(predictions[0][idx]) * 100
        class_name = classes[idx] if idx < len(classes) else f"Class_{idx}"
        results[rank] = {
            'class': class_name,
            'confidence': confidence,
            'class_id': int(idx)
        }
    
    return results, img

# ============================================================================
# 4. DEMO PREDICTIONS
# ============================================================================
print("\n3. Making predictions on sample images...")

# Create demo image from dataset if available
demo_dir = "d:/task4/food-101/images"
demo_images = []

if os.path.exists(demo_dir):
    # Find first few food category directories
    for category in os.listdir(demo_dir)[:3]:
        category_path = os.path.join(demo_dir, category)
        if os.path.isdir(category_path):
            # Get first image from category
            images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
            if images:
                demo_images.append(os.path.join(category_path, images[0]))

# Make predictions
if demo_images:
    fig, axes = plt.subplots(1, len(demo_images), figsize=(15, 5))
    if len(demo_images) == 1:
        axes = [axes]
    
    fig.suptitle('Food-101 Predictions', fontsize=14, fontweight='bold')
    
    for idx, img_path in enumerate(demo_images):
        print(f"\n   Image {idx+1}: {os.path.basename(img_path)}")
        
        results, img = predict_image(img_path, top_k=3)
        
        if results:
            for rank, pred in results.items():
                print(f"      {rank}. {pred['class']}: {pred['confidence']:.2f}%")
            
            # Plot image with predictions
            ax = axes[idx]
            ax.imshow(img)
            title_text = f"{results[1]['class']}\n{results[1]['confidence']:.1f}%"
            ax.set_title(title_text, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_demo.png', dpi=100, bbox_inches='tight')
    print("\n   ✓ Saved: predictions_demo.png")
    plt.close()
else:
    print("   ⚠ No sample images found to demonstrate predictions")

# ============================================================================
# 5. PREDICTION ON CUSTOM IMAGE
# ============================================================================
print("\n4. Function for custom predictions:")
print("""
   Example usage:
   
   results, img = predict_image('path/to/your/image.jpg', top_k=5)
   
   if results:
       for rank, pred in results.items():
           print(f"{rank}. {pred['class']}: {pred['confidence']:.2f}%")
""")

print("\n" + "=" * 60)
print("Prediction tool ready!")
print("=" * 60)
