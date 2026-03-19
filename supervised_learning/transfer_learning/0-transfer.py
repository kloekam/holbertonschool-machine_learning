#!/usr/bin/env python3
"""
Stanford Dogs Dataset Classifier using Transfer Learning (EfficientNetB3).
Trains a CNN to classify 120 dog breeds with >=87% validation accuracy.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


# Data loading

def load_stanford_dogs(img_size: int = 300, batch_size: int = 32):
    """
    Downloads and loads the Stanford Dogs dataset via tensorflow_datasets.
    Returns (ds_train, ds_val) as tf.data.Dataset pipelines.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        return image, label

    def preprocess(sample):
        image = tf.cast(sample["image"], tf.float32)
        image = tf.image.resize(image, (img_size, img_size))
        image = keras.applications.efficientnet.preprocess_input(image)
        label = tf.one_hot(sample["label"], 120)
        return image, label

    (ds_train_raw, ds_val_raw), _ = tfds.load(
        "stanford_dogs",
        split=["train", "test"],
        as_supervised=False,
        with_info=True,
        shuffle_files=True,
    )

    ds_train = (
        ds_train_raw
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(2048, seed=42)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    ds_val = (
        ds_val_raw
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return ds_train, ds_val


# Model

def build_model(num_classes: int = 120):
    """
    Builds an EfficientNetB3 transfer-learning model.
    The first layer is a Lambda that resizes inputs to 300x300,
    matching EfficientNetB3's canonical input resolution.
    Returns (model, base) so the base can be selectively unfrozen later.
    """
    inputs = keras.Input(shape=(None, None, 3), name="image_input")

    # Lambda layer
    x = layers.Lambda(
        lambda img: tf.image.resize(img, (300, 300)),
        name="resize_to_300"
    )(inputs)

    # EfficientNetB3 base
    base = keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(300, 300, 3),
    )
    base.trainable = False

    x = base(x, training=False)

    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="stanford_dogs_efficientnetb3")
    return model, base


# Training

def train():
    # Load data
    ds_train, ds_val = load_stanford_dogs(img_size=300)

    # Build model
    print("Building model ...")
    model, base = build_model(num_classes=120)
    model.summary(line_length=100)

    # Phase 1 – train head only (base frozen)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=20,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5,
                                          restore_best_weights=True,
                                          monitor="val_accuracy"),
            keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                              patience=3,
                                              monitor="val_accuracy",
                                              verbose=1),
        ],
    )

    # Phase 2 – fine-tune top 60 layers of base
    base.trainable = True
    for layer in base.layers[:-60]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=50,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=7,
                                          restore_best_weights=True,
                                          monitor="val_accuracy"),
            keras.callbacks.ReduceLROnPlateau(factor=0.3,
                                              patience=3,
                                              monitor="val_accuracy",
                                              verbose=1),
            keras.callbacks.ModelCheckpoint(
                "stanford_dogs_best.h5",
                save_best_only=True,
                monitor="val_accuracy",
                verbose=1,
            ),
        ],
    )

    # Evaluate & save
    val_loss, val_acc = model.evaluate(ds_val)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")

    if val_acc < 0.87:
        print("Validation accuracy below 87% target.")

    model.save("stanford_dogs.h5")
    print("Model saved to stanford_dogs.h5")

    return history


if __name__ == "__main__":
    train()
