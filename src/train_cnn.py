"""Treina uma CNN de gêneros musicais sobre espectrogramas mel.

Os espectrogramas (features/cnn/X_segments.npy) são segmentos de 3s de cada
track, 10 segmentos por track. A divisão treino/validação/teste é feita por
TRACK (não por segmento) para evitar vazamento de segmentos da mesma música
entre os conjuntos. A avaliação é reportada em dois níveis: por segmento e por
track (média das probabilidades softmax dos 10 segmentos).

Arquitetura: duas camadas convolucionais com ReLU, max pooling, dropout e
camadas totalmente conectadas para classificação.

Uso:
    python src/train_cnn.py [--epochs N] [--max-tracks N]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT_DIR / "features" / "cnn"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

RANDOM_STATE = 42
SEGMENTS_PER_TRACK = 10
TEST_SIZE = 0.2  # fração de tracks no teste
VAL_SIZE = 0.125  # fração do restante (~10% do total)
BATCH_SIZE = 32

# espectrogramas estão em dB relativos ao máximo do segmento (~[-80, 0]);
# mapeia linearmente para ~[0, 1]
DB_OFFSET = 80.0


def load_data():
    # memmap: o arquivo tem ~865 MB, evita carregar tudo na RAM
    X = np.load(FEATURES_DIR / "X_segments.npy", mmap_mode="r")
    y = np.load(FEATURES_DIR / "y_segments.npy")
    classes = np.load(FEATURES_DIR / "classes.npy")

    track_ids_path = FEATURES_DIR / "track_ids.npy"
    if track_ids_path.exists():
        track_ids = np.load(track_ids_path)
    else:
        # os segmentos foram gravados sequencialmente, 10 por track
        track_ids = np.arange(len(y)) // SEGMENTS_PER_TRACK

    return X, y, classes, track_ids


def split_by_track(y, track_ids, max_tracks=None):
    """Divide os índices de segmentos em treino/val/teste estratificando por track."""
    unique_tracks = np.unique(track_ids)
    # label de cada track = label do seu primeiro segmento
    first_seg = np.searchsorted(track_ids, unique_tracks)
    track_labels = y[first_seg]

    if max_tracks is not None and max_tracks < len(unique_tracks):
        unique_tracks, _, track_labels, _ = train_test_split(
            unique_tracks,
            track_labels,
            train_size=max_tracks,
            stratify=track_labels,
            random_state=RANDOM_STATE,
        )

    train_val_tracks, test_tracks, train_val_labels, _ = train_test_split(
        unique_tracks,
        track_labels,
        test_size=TEST_SIZE,
        stratify=track_labels,
        random_state=RANDOM_STATE,
    )
    train_tracks, val_tracks = train_test_split(
        train_val_tracks,
        test_size=VAL_SIZE,
        stratify=train_val_labels,
        random_state=RANDOM_STATE,
    )

    def segment_indices(tracks):
        return np.flatnonzero(np.isin(track_ids, tracks))

    return (
        segment_indices(train_tracks),
        segment_indices(val_tracks),
        segment_indices(test_tracks),
    )


def make_dataset(X, y, indices, input_shape, shuffle=False):
    """Cria um tf.data.Dataset que lê os espectrogramas do memmap por batch."""

    def fetch(batch_idx):
        # ordena os índices para leitura sequencial do memmap (mais rápida)
        batch_idx = np.sort(batch_idx)
        xb = (X[batch_idx].astype(np.float32) + DB_OFFSET) / DB_OFFSET
        return xb, y[batch_idx]

    def map_fn(batch_idx):
        xb, yb = tf.numpy_function(fetch, [batch_idx], (tf.float32, tf.int32))
        xb.set_shape((None, *input_shape))
        yb.set_shape((None,))
        return xb, yb

    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(len(indices), seed=RANDOM_STATE, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

"""nos baseamos no paper https://arxiv.org/html/2509.01762v1"""
def build_model(input_shape, n_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history, path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    ax_loss.plot(history.history["loss"], label="treino")
    ax_loss.plot(history.history["val_loss"], label="validação")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(history.history["accuracy"], label="treino")
    ax_acc.plot(history.history["val_accuracy"], label="validação")
    ax_acc.set_xlabel("Época")
    ax_acc.set_ylabel("Acurácia")
    ax_acc.set_title("Acurácia")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Usa apenas N tracks (para testes rápidos)",
    )
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    X, y, classes, track_ids = load_data()
    input_shape = X.shape[1:]
    print(f"Dados: X {X.shape}, y {y.shape}, {len(classes)} gêneros")
    print(f"Gêneros: {list(classes)}\n")

    train_idx, val_idx, test_idx = split_by_track(y, track_ids, args.max_tracks)
    print(
        f"Segmentos - treino: {len(train_idx)} | validação: {len(val_idx)} | "
        f"teste: {len(test_idx)}\n"
    )

    train_ds = make_dataset(X, y, train_idx, input_shape, shuffle=True)
    val_ds = make_dataset(X, y, val_idx, input_shape)
    test_ds = make_dataset(X, y, test_idx, input_shape)

    model = build_model(input_shape, len(classes))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]

    print("\nTreinando...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curves_path = RESULTS_DIR / "cnn_training_curves.png"
    plot_training_curves(history, curves_path)
    print(f"\nCurvas de treino salvas em {curves_path}")

    # ---- avaliação por segmento ----
    probs = model.predict(test_ds, verbose=0)
    y_test = y[test_idx]
    y_pred_seg = probs.argmax(axis=1)
    seg_accuracy = accuracy_score(y_test, y_pred_seg)

    print(f"\nAcurácia no teste (por segmento): {seg_accuracy:.4f}\n")
    print("Relatório por gênero (segmentos):")
    print(
        classification_report(
            y_test, y_pred_seg, target_names=classes, zero_division=0
        )
    )

    # ---- avaliação por track: média das probabilidades dos 10 segmentos ----
    test_track_ids = track_ids[test_idx]
    unique_test_tracks = np.unique(test_track_ids)
    track_probs = np.stack(
        [probs[test_track_ids == t].mean(axis=0) for t in unique_test_tracks]
    )
    y_true_track = np.array(
        [y_test[test_track_ids == t][0] for t in unique_test_tracks]
    )
    y_pred_track = track_probs.argmax(axis=1)
    track_accuracy = accuracy_score(y_true_track, y_pred_track)

    print(f"Acurácia no teste (por track): {track_accuracy:.4f}\n")
    print("Relatório por gênero (tracks):")
    print(
        classification_report(
            y_true_track, y_pred_track, target_names=classes, zero_division=0
        )
    )

    cm = confusion_matrix(y_true_track, y_pred_track)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"CNN - Matriz de Confusão por track (acurácia teste: {track_accuracy:.2%})")
    fig.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix_cnn.png"
    fig.savefig(cm_path, dpi=150)
    print(f"Matriz de confusão salva em {cm_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "cnn_model.keras"
    model.save(model_path)
    print(f"Modelo salvo em {model_path}")


if __name__ == "__main__":
    main()
