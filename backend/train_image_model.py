"""
VetAI — Image-based disease detection trainer (MobileNetV2, transfer learning).

Turns the Roboflow COCO export into a disease classifier for: healthy, fmd, lsd.

Usage:
    python train_image_model.py /path/to/unzipped/roboflow/dataset

It prints a class summary first, trains, evaluates, and saves:
    app/ml/image_model.keras   (the trained model)
    app/ml/image_classes.json  (class order + metrics)
"""

import json
import os
import sys
import numpy as np
import tensorflow as tf

from app.ml.image_data import (
    IMG_SIZE, DISEASE_CLASSES, find_splits, load_split, summarize,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT = os.path.join(HERE, "app", "ml", "image_model.keras")
CLASSES_OUT = os.path.join(HERE, "app", "ml", "image_classes.json")

BATCH = 32
EPOCHS = 12
AUTOTUNE = tf.data.AUTOTUNE


def _decode(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label


def make_ds(filepaths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(len(filepaths))
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)


def build_model(n_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main(dataset_dir):
    print("== Dataset summary ==")
    summarize(dataset_dir)

    splits = find_splits(dataset_dir)
    train_dir = splits.get("train")
    if not train_dir:
        print("Could not find a training split. Aborting.")
        return

    tr_fp, tr_lab, _ = load_split(train_dir)
    if "valid" in splits:
        va_fp, va_lab, _ = load_split(splits["valid"])
    else:
        # carve a validation set from training data
        idx = int(len(tr_fp) * 0.85)
        tr_fp, va_fp = tr_fp[:idx], tr_fp[idx:]
        tr_lab, va_lab = tr_lab[:idx], tr_lab[idx:]

    print(f"\nTraining on {len(tr_fp)} images, validating on {len(va_fp)}.")

    # class weights (datasets are imbalanced)
    counts = np.bincount(tr_lab, minlength=len(DISEASE_CLASSES))
    total = counts.sum()
    class_weight = {i: (total / (len(DISEASE_CLASSES) * c)) if c else 0.0
                    for i, c in enumerate(counts)}

    train_ds = make_ds(tr_fp, tr_lab, training=True)
    val_ds = make_ds(va_fp, va_lab)

    model = build_model(len(DISEASE_CLASSES))
    early = tf.keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor="val_accuracy")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              class_weight=class_weight, callbacks=[early])

    # evaluate on test split if present
    metrics = {}
    if "test" in splits:
        te_fp, te_lab, _ = load_split(splits["test"])
        if te_fp:
            loss, acc = model.evaluate(make_ds(te_fp, te_lab), verbose=0)
            metrics["test_accuracy"] = float(acc)
            print(f"\nTest accuracy: {acc:.3f}")

    model.save(MODEL_OUT)
    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        json.dump({"classes": DISEASE_CLASSES, "metrics": metrics}, f, indent=2)
    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved classes -> {CLASSES_OUT}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_image_model.py /path/to/dataset")
        sys.exit(1)
    main(sys.argv[1])
