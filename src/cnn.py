import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42
SEGMENTS_PER_TRACK = 10
TEST_SIZE = 0.2
VAL_SIZE = 0.125
BATCH_SIZE = 32

ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT_DIR / "features" / "cnn"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

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
def make_dataset(X, y, indices, input_shape, shuffle=False):
    # fetch reads ONE batch from the memmap and normalizes it
    def fetch(batch_idx):
        batch_idx = np.sort(batch_idx)               # sequential memmap reads are faster
        xb = (X[batch_idx].astype(np.float32) + DB_OFFSET) / DB_OFFSET
        return xb, y[batch_idx]

    # tf.numpy_function lets TensorFlow call the numpy fetch on a batch
    def map_fn(batch_idx):
        xb, yb = tf.numpy_function(fetch, [batch_idx], (tf.float32, tf.int32))
        xb.set_shape((None, *input_shape))           # TensorFlow needs static shapes
        yb.set_shape((None,))
        return xb, yb

    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(len(indices), seed=RANDOM_STATE, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def build_model(input_shape, n_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape))) # input_shape = (128, 130, 1)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

def main():
    X, y, classes, track_ids = load_data()
    input_shape = X.shape[1:]                     # (128, 130, 1) derivado dos dados
    print(f"Dados: X {X.shape}, y {y.shape}, {len(classes)} gêneros")

    train_idx, val_idx, test_idx = split_by_track(y, track_ids)
    print(f"Segmentos - treino: {len(train_idx)} | validação: {len(val_idx)} | teste: {len(test_idx)}")

    train_ds = make_dataset(X, y, train_idx, input_shape, shuffle=True)
    val_ds   = make_dataset(X, y, val_idx, input_shape)
    test_ds  = make_dataset(X, y, test_idx, input_shape)

    model = build_model(input_shape, len(classes))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=60,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- avaliação por segmento ----
    probs = model.predict(test_ds, verbose=0)
    y_test = y[test_idx]                 # rótulos reais dos segmentos de teste
    y_pred_seg = probs.argmax(axis=1)    # classe com maior probabilidade, por segmento
    seg_accuracy = accuracy_score(y_test, y_pred_seg)
    print(f"Acurácia (por segmento): {seg_accuracy:.4f}")
    print(classification_report(y_test, y_pred_seg, target_names=classes, zero_division=0))

    # ---- avaliação por track: média das probabilidades softmax dos 10 segmentos ----
    test_track_ids = track_ids[test_idx]
    unique_test_tracks = np.unique(test_track_ids)
    track_probs = np.stack(
        [probs[test_track_ids == t].mean(axis=0) for t in unique_test_tracks]
    )
    y_true_track = np.array([y_test[test_track_ids == t][0] for t in unique_test_tracks])
    y_pred_track = track_probs.argmax(axis=1)
    track_accuracy = accuracy_score(y_true_track, y_pred_track)
    print(f"Acurácia (por track): {track_accuracy:.4f}")
    print(classification_report(y_true_track, y_pred_track, target_names=classes, zero_division=0))

    # ---- salvar ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true_track, y_pred_track)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"CNN - Matriz de Confusão por track (teste: {track_accuracy:.2%})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix_cnn.png", dpi=150)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "cnn_model.keras")
    print("Modelo salvo em models/cnn_model.keras")

if __name__ == "__main__":
    main()

