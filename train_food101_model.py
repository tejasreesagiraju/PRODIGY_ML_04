"""
Food-101 Model Training Script
Train a transfer learning model on the Food-101 dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# Configuration
DATASET_PATH = "d:/task4"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10
LEARNING_RATE = 1e-4

print("=" * 60)
print("FOOD-101 MODEL TRAINING")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading and preparing data...")

images_path = os.path.join(DATASET_PATH, "food-101", "images")

if os.path.exists(images_path):
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
    print("   ✓ Data loaded from local directory")
else:
    print("   ⚠ Local dataset not found, creating demo dataset")
    train_ds = tf.data.Dataset.from_tensor_slices((
        np.random.rand(1000, 224, 224, 3),
        tf.one_hot(np.random.randint(0, 101, 1000), 101)
    )).batch(BATCH_SIZE).shuffle(1000)
    
    val_ds = tf.data.Dataset.from_tensor_slices((
        np.random.rand(200, 224, 224, 3),
        tf.one_hot(np.random.randint(0, 101, 200), 101)
    )).batch(BATCH_SIZE)

# Normalize images
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

print(f"   ✓ Training batches: {len(train_ds)}")
print(f"   ✓ Validation batches: {len(val_ds)}")

# Get number of classes
num_classes = next(iter(train_ds))[1].shape[1]
print(f"   ✓ Number of classes: {num_classes}")

# ============================================================================
# 2. CREATE MODEL WITH TRANSFER LEARNING
# ============================================================================
print("\n2. Creating transfer learning model...")
print("   Using MobileNetV2 as base model")

# Load pre-trained MobileNetV2
base_model = keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
base_model.trainable = False

# Create new model
model = keras.Sequential([
    keras.layers.Input(shape=IMAGE_SIZE + (3,)),
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

print("   ✓ Model created")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# ============================================================================
# 3. CALLBACKS
# ============================================================================
print("\n3. Setting up callbacks...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'food101_best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("   ✓ Callbacks configured")

# ============================================================================
# 4. TRAIN MODEL
# ============================================================================
print("\n4. Training model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print("   " + "-" * 56)

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n   Training completed in {training_time:.2f} seconds")

# ============================================================================
# 5. PLOT TRAINING HISTORY
# ============================================================================
print("\n5. Plotting training history...")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: training_history.png")
plt.close()

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n6. Evaluating model on validation set...")

val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print(f"   • Validation Loss: {val_loss:.4f}")
print(f"   • Validation Accuracy: {val_accuracy:.4f}")

# ============================================================================
# 7. SAVE MODEL
# ============================================================================
print("\n7. Saving model...")

model.save('food101_model.h5')
print("   ✓ Model saved: food101_model.h5")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
print("\nGenerated files:")
print("   • food101_best_model.h5 - Best model checkpoint")
print("   • food101_model.h5 - Final trained model")
print("   • training_history.png - Training plots")
