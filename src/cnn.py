import numpy as np
from pathlib import Path

RANDOM_STATE = 42
SEGMENTS_PER_TRACK = 10
TEST_SIZE = 0.2
VAL_SIZE = 0.125
BATCH_SIZE = 32

FEATURES_DIR = Path(__file__).resolve().parent.parent / "features" / "cnn"
DB_OFFSET = 80.0

# 1) load X, y, classes from .npy

#mmap to avoid loading 800MB into RAM at the same time
X = np.load(FEATURES_DIR / "X_segments.npy", mmap_mode="r") 
y = np.load(FEATURES_DIR / "y_segments.npy")
classes = np.load(FEATURES_DIR / "classes.npy")

import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(128, 130, 1)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(13, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
