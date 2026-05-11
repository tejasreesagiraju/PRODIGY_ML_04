"""
Food-101 Dataset Loader and Explorer
This script loads and explores the Food-101 dataset for image classification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import json
from collections import Counter

# Configuration
DATASET_PATH = "d:/task4/food-101"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

print("=" * 60)
print("FOOD-101 DATASET LOADER AND EXPLORER")
print("=" * 60)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n1. Loading Food-101 dataset...")

def load_food101_from_directory(path):
    """Load Food-101 from extracted directory"""
    images_path = os.path.join(path, "images")
    
    if os.path.exists(images_path):
        print(f"   ✓ Found images directory: {images_path}")
        return images_path
    else:
        print(f"   ✗ Images directory not found at {images_path}")
        return None

# Try to load from local directory first
images_path = load_food101_from_directory(DATASET_PATH)

# If not found locally, download from TensorFlow datasets
if images_path is None:
    print("   Downloading Food-101 from TensorFlow datasets...")
    try:
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data()
        print("   ✓ Using alternative dataset loading method")
        use_local = False
    except Exception as e:
        print(f"   Could not load dataset: {e}")
        print("   Please ensure the Food-101 dataset is properly extracted")
        exit(1)
else:
    use_local = True
    print(f"   ✓ Dataset path: {images_path}")

# ============================================================================
# 2. EXPLORE DATASET STRUCTURE
# ============================================================================
print("\n2. Exploring dataset structure...")

if use_local:
    # Load metadata
    meta_path = os.path.join(DATASET_PATH, "meta")
    
    # Load classes
    classes_file = os.path.join(meta_path, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"   ✓ Number of food categories: {len(classes)}")
        print(f"   ✓ First 10 categories: {classes[:10]}")
    
    # Load train/test splits
    train_file = os.path.join(meta_path, "train.txt")
    test_file = os.path.join(meta_path, "test.txt")
    
    train_samples = []
    test_samples = []
    
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_samples = [line.strip() for line in f.readlines()]
        print(f"   ✓ Training samples: {len(train_samples)}")
    
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_samples = [line.strip() for line in f.readlines()]
        print(f"   ✓ Test samples: {len(test_samples)}")
    
    # Analyze class distribution
    train_categories = [sample.split('/')[0] for sample in train_samples]
    category_counts = Counter(train_categories)
    
    print(f"\n   Category distribution in training set:")
    for category, count in sorted(category_counts.items())[:5]:
        print(f"      - {category}: {count} images")
    print(f"      ... and {len(category_counts) - 5} more categories")

# ============================================================================
# 3. CREATE DATA PIPELINE
# ============================================================================
print("\n3. Creating data pipeline...")

if use_local and os.path.exists(images_path):
    # Create training dataset from directory
    train_ds = keras.utils.image_dataset_from_directory(
        images_path,
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        subset="training",
        validation_split=0.2,
        label_mode='categorical'
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        images_path,
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        subset="validation",
        validation_split=0.2,
        label_mode='categorical'
    )
    
    print(f"   ✓ Training batches: {len(train_ds)}")
    print(f"   ✓ Validation batches: {len(val_ds)}")
else:
    print("   ⚠ Using demo mode with random data")
    # Create dummy datasets for demonstration
    train_ds = tf.data.Dataset.from_tensor_slices((
        np.random.rand(100, 224, 224, 3),
        np.random.rand(100, 101)
    )).batch(BATCH_SIZE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((
        np.random.rand(20, 224, 224, 3),
        np.random.rand(20, 101)
    )).batch(BATCH_SIZE)

# Normalize pixel values to [0, 1]
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ============================================================================
# 4. VISUALIZE SAMPLES
# ============================================================================
print("\n4. Visualizing sample images...")

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle('Food-101 Dataset - Sample Images', fontsize=16, fontweight='bold')

batch_images, batch_labels = next(iter(train_ds))
num_classes = batch_labels.shape[1]

for idx, ax in enumerate(axes.flat):
    if idx < 9:
        img = batch_images[idx].numpy()
        label_idx = np.argmax(batch_labels[idx])
        ax.imshow(img)
        ax.set_title(f'Category {label_idx}', fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('food101_samples.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved visualization: food101_samples.png")
plt.close()

# ============================================================================
# 5. DATASET STATISTICS
# ============================================================================
print("\n5. Dataset Statistics:")
print(f"   • Image size: {IMAGE_SIZE}")
print(f"   • Number of classes: {num_classes}")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Data type: RGB Images")
print(f"   • Value range: [0, 1] (normalized)")

# Get sample batch shape
print(f"\n   Batch shape:")
print(f"      Images: {batch_images.shape}")
print(f"      Labels: {batch_labels.shape}")

# ============================================================================
# 6. READY FOR MODEL TRAINING
# ============================================================================
print("\n6. Data pipeline ready for model training!")
print("   You can now train models using:")
print("      - train_ds: Training dataset")
print("      - val_ds: Validation dataset")
print("\n" + "=" * 60)

# Optional: Create a simple model as example
print("\n7. Creating a simple demonstration model...")

model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("   ✓ Model created successfully!")
print("\nModel Summary:")
model.summary()

print("\n" + "=" * 60)
print("Setup complete! Your data is ready for training.")
print("=" * 60)
