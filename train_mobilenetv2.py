# train_mobilenetv2.py
"""
MobileNetV2 transfer learning pipeline for dog-emotion classification.
Optimized version with:
- Correct class order
- Data augmentation (train only)
- Preprocessing using preprocess_input
- Fine-tuning
- Class weights
- Callbacks
- ONNX export
"""

import os
import json
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tf2onnx

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_DIR = "dataset"
OUTPUT_DIR = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Updated hyperparameters (optimized)
NUM_EPOCHS_HEAD = 10
NUM_EPOCHS_FINE = 25
FINE_TUNE_AT = 90

LR_HEAD = 1e-3
LR_FINE = 1e-5

PATIENCE = 5
AUTOTUNE = tf.data.AUTOTUNE

# CLASS ORDER MUST MATCH DATASET SUBFOLDER ORDER
CLASS_LABELS = ["angry", "happy", "relaxed", "sad"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


# ------------------------------------------------------------
# BUILD TRAIN/VAL LOADER (CORRECTED)
# ------------------------------------------------------------
def build_train_val(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.15, seed=1337):

    # Train dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_LABELS,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        validation_split=val_split,
        subset="training",
        seed=seed
    )

    # Validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_LABELS,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        validation_split=val_split,
        subset="validation",
        seed=seed
    )

    # TRAIN: augment → preprocess_input
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(tf.cast(data_augmentation(x, training=True), tf.float32)), y),
        num_parallel_calls=AUTOTUNE
    )

    # VALIDATION: only preprocess_input
    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
        num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds


# ------------------------------------------------------------
# CLASS WEIGHTS
# ------------------------------------------------------------
def compute_class_weights_from_dir(data_dir, class_names=CLASS_LABELS):
    counts = {}
    for cls in class_names:
        p = Path(data_dir) / cls
        counts[cls] = len(list(p.glob("*")))

    total = sum(counts.values())
    class_weights = {
        i: total / (len(class_names) * counts[cls])
        for i, cls in enumerate(class_names)
    }

    print("Class counts:", counts)
    print("Class weights:", class_weights)

    return class_weights


# ------------------------------------------------------------
# BUILD MODEL
# ------------------------------------------------------------
def build_model(num_classes=len(CLASS_LABELS), input_shape=IMG_SIZE + (3,)):
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model, base


# ------------------------------------------------------------
# TRAIN PIPELINE
# ------------------------------------------------------------
def train(data_dir):

    train_ds, val_ds = build_train_val(data_dir)
    class_weights = compute_class_weights_from_dir(data_dir)

    model, base = build_model()

    # HEAD TRAINING
    model.compile(
        optimizer=optimizers.Adam(LR_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_head = os.path.join(OUTPUT_DIR, "best_head.h5")
    cb_head = [
        callbacks.ModelCheckpoint(ckpt_head, save_best_only=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(patience=3),
        callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
    ]

    print("\n----------------------")
    print("Training model head...")
    print("----------------------")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=cb_head,
    )

    # FINE-TUNING
    print("\n----------------------")
    print("Fine tuning...")
    print("----------------------")

    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(LR_FINE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_ft = os.path.join(OUTPUT_DIR, "best_finetuned.h5")
    cb_ft = [
        callbacks.ModelCheckpoint(ckpt_ft, save_best_only=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(patience=3),
        callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS_FINE,
        class_weight=class_weights,
        callbacks=cb_ft,
    )

    # SAVE MODEL
    final_h5 = os.path.join(OUTPUT_DIR, "mobilenetv2_final.h5")
    model.save(final_h5)
    print("\nSaved final model:", final_h5)

    # SAVE LABELS
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump(CLASS_LABELS, f)

    # ONNX EXPORT
    try:
        onnx_path = os.path.join(OUTPUT_DIR, "mobilenetv2_final.onnx")
        spec = (tf.TensorSpec((None, IMG_SIZE[0], IMG_SIZE[1], 3), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print("Saved ONNX model to:", onnx_path)
    except Exception as e:
        print("ONNX export failed:", e)

    loss, acc = model.evaluate(val_ds)
    print(f"\nValidation Loss: {loss:.4f}  |  Accuracy: {acc:.4f}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    train(DATA_DIR)

