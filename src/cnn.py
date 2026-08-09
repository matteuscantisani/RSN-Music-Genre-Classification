import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
SEGMENTS_PER_TRACK = 10
TEST_SIZE = 0.2
VAL_SIZE = 0.125
BATCH_SIZE = 32

FEATURES_DIR = Path(__file__).resolve().parent.parent / "features" / "cnn"
DB_OFFSET = 80.0

#load X, y, classes from .npy
def load_data():
    #mmap to avoid loading 800MB into RAM at the same time
    X = np.load(FEATURES_DIR / "X_segments.npy", mmap_mode="r")
    y = np.load(FEATURES_DIR / "y_segments.npy")
    classes = np.load(FEATURES_DIR / "classes.npy")

    track_ids = np.arange(len(y)) // SEGMENTS_PER_TRACK

    return X, y, classes, track_ids

def split_by_track(y, track_ids, max_tracks=None):
    # each unique track is an indivisible unit
    unique_tracks = np.unique(track_ids)
    # label of each track = label of its first segment (segments written in order)
    first_seg = np.searchsorted(track_ids, unique_tracks)
    track_labels = y[first_seg]

    # split 1: train+val vs test (80/20), stratified by genre
    train_val_tracks, test_tracks, train_val_labels, _ = train_test_split(
        unique_tracks, track_labels,
        test_size=TEST_SIZE, stratify=track_labels, random_state=RANDOM_STATE,
    )
    # split 2: train vs val (the 80% becomes 70/10)
    train_tracks, val_tracks = train_test_split(
        train_val_tracks,
        test_size=VAL_SIZE, stratify=train_val_labels, random_state=RANDOM_STATE,
    )

    def segment_indices(tracks):
        return np.flatnonzero(np.isin(track_ids, tracks))

    return (
        segment_indices(train_tracks),
        segment_indices(val_tracks),
        segment_indices(test_tracks),
    )



def build_model():
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

X, y, classes, track_ids = load_data()
print(split_by_track(y, track_ids))

