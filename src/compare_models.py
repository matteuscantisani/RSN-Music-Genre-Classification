"""Compara SVM e CNN com intervalos de confiança, sem retreinar os modelos.

Carrega os modelos salvos em models/, reconstrói o split de teste (seed 42,
o mesmo usado nos dois treinos) e calcula, via bootstrap (scipy), IC de 95%
para a acurácia de cada modelo e para a diferença pareada entre eles, além
do teste exato de McNemar.

Uso:
    python src/compare_models.py

Saídas:
    results/comparison_stats.txt
    results/comparison_ci.png
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import train_test_split

from train_cnn import (
    RANDOM_STATE,
    TEST_SIZE,
    load_data as load_cnn_data,
    make_dataset,
    split_by_track,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SVM_FEATURES_DIR = ROOT_DIR / "features" / "svm"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95


def svm_correct_per_track():
    """Vetor 0/1 de acertos do SVM nos tracks de teste (ordenado por track id)."""
    X = np.load(SVM_FEATURES_DIR / "X.npy")
    y = np.load(SVM_FEATURES_DIR / "y.npy")

    # mesmo split de train_svm.py: dividir um vetor de índices com a mesma
    # seed/estratificação reproduz exatamente a mesma partição
    _, test_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    test_idx = np.sort(test_idx)

    model = joblib.load(MODELS_DIR / "svm_model.joblib")
    y_pred = model.predict(X[test_idx])

    return (y_pred == y[test_idx]).astype(int), test_idx


def cnn_correct_per_track():
    """Vetor 0/1 de acertos da CNN por track de teste (ordenado por track id)."""
    X, y, _, track_ids = load_cnn_data()
    input_shape = X.shape[1:]

    _, _, test_idx = split_by_track(y, track_ids)
    test_ds = make_dataset(X, y, test_idx, input_shape)

    model = tf.keras.models.load_model(MODELS_DIR / "cnn_model.keras")
    probs = model.predict(test_ds, verbose=0)

    # agrega por track: média das probabilidades softmax dos 10 segmentos
    test_track_ids = track_ids[test_idx]
    y_test = y[test_idx]
    unique_tracks = np.unique(test_track_ids)
    track_probs = np.stack(
        [probs[test_track_ids == t].mean(axis=0) for t in unique_tracks]
    )
    y_true = np.array([y_test[test_track_ids == t][0] for t in unique_tracks])
    y_pred = track_probs.argmax(axis=1)

    return (y_pred == y_true).astype(int), unique_tracks


def ci_mean_bootstrap(sample, rng):
    """IC 95% (BCa) para a média de um vetor via bootstrap."""
    res = stats.bootstrap(
        (sample,),
        np.mean,
        n_resamples=N_RESAMPLES,
        confidence_level=CONFIDENCE_LEVEL,
        method="BCa",
        random_state=rng,
    )
    return res.confidence_interval, res.bootstrap_distribution


def main():
    print("Avaliando SVM no conjunto de teste...")
    svm_correct, svm_tracks = svm_correct_per_track()
    print("Avaliando CNN no conjunto de teste...")
    cnn_correct, cnn_tracks = cnn_correct_per_track()

    # garantia de que os dois modelos foram avaliados nos MESMOS tracks
    assert np.array_equal(svm_tracks, cnn_tracks), (
        "Os conjuntos de teste do SVM e da CNN não coincidem!"
    )
    n_test = len(svm_tracks)

    svm_acc = svm_correct.mean()
    cnn_acc = cnn_correct.mean()
    print(f"\nTracks de teste: {n_test}")
    print(f"Acurácia SVM: {svm_acc:.4f}")
    print(f"Acurácia CNN: {cnn_acc:.4f}")

    rng = np.random.default_rng(RANDOM_STATE)
    svm_ci, _ = ci_mean_bootstrap(svm_correct, rng)
    cnn_ci, _ = ci_mean_bootstrap(cnn_correct, rng)

    # bootstrap pareado da diferença: reamostrar o vetor de diferenças por
    # track equivale a reamostrar os mesmos tracks para os dois modelos
    diff = cnn_correct.astype(float) - svm_correct.astype(float)
    diff_ci, diff_dist = ci_mean_bootstrap(diff, rng)
    diff_mean = diff.mean()

    # teste exato de McNemar sobre os pares discordantes
    b = int(np.sum((cnn_correct == 1) & (svm_correct == 0)))  # só CNN acerta
    c = int(np.sum((cnn_correct == 0) & (svm_correct == 1)))  # só SVM acerta
    mcnemar_p = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0

    ci_excludes_zero = diff_ci.low > 0 or diff_ci.high < 0
    mcnemar_significant = mcnemar_p < 0.05
    better = "CNN" if diff_mean > 0 else "SVM"
    if ci_excludes_zero and mcnemar_significant:
        conclusion = (
            f"A diferença é estatisticamente significativa (IC 95% não contém 0 "
            f"e McNemar p = {mcnemar_p:.4g} < 0.05): {better} é superior neste teste."
        )
    elif ci_excludes_zero or mcnemar_significant:
        conclusion = (
            f"Resultado limítrofe: os dois testes discordam (IC 95% "
            f"{'exclui' if ci_excludes_zero else 'contém'} 0; McNemar p = "
            f"{mcnemar_p:.4g}). Há evidência fraca de que {better} seja superior, "
            f"mas não conclusiva ao nível de 95%."
        )
    else:
        conclusion = (
            f"A diferença NÃO é estatisticamente significativa ao nível de 95% "
            f"(IC contém 0; McNemar p = {mcnemar_p:.4g})."
        )

    lines = [
        "Comparação SVM vs CNN — modelos salvos, mesmo conjunto de teste",
        f"(bootstrap BCa, {N_RESAMPLES} reamostragens, IC de {CONFIDENCE_LEVEL:.0%})",
        "",
        f"Tracks de teste: {n_test} (split estratificado, seed {RANDOM_STATE})",
        "",
        f"SVM: acurácia = {svm_acc:.4f}  IC 95% = [{svm_ci.low:.4f}, {svm_ci.high:.4f}]",
        f"CNN: acurácia = {cnn_acc:.4f}  IC 95% = [{cnn_ci.low:.4f}, {cnn_ci.high:.4f}]",
        "",
        f"Diferença pareada (CNN - SVM): {diff_mean:+.4f}  "
        f"IC 95% = [{diff_ci.low:+.4f}, {diff_ci.high:+.4f}]",
        f"Pares discordantes: só CNN acerta = {b}, só SVM acerta = {c}",
        f"Teste de McNemar (exato): p = {mcnemar_p:.4g}",
        "",
        conclusion,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS_DIR / "comparison_stats.txt"
    stats_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nEstatísticas salvas em {stats_path}")

    # ---- gráfico: acurácias com IC + distribuição bootstrap da diferença ----
    fig, (ax_acc, ax_diff) = plt.subplots(1, 2, figsize=(12, 5))

    names = ["SVM", "CNN"]
    accs = [svm_acc, cnn_acc]
    err_low = [svm_acc - svm_ci.low, cnn_acc - cnn_ci.low]
    err_high = [svm_ci.high - svm_acc, cnn_ci.high - cnn_acc]
    ax_acc.bar(names, accs, color=["#4c72b0", "#dd8452"], width=0.5)
    ax_acc.errorbar(
        names, accs, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=6
    )
    for i, (acc, lo, hi) in enumerate(
        [(svm_acc, svm_ci.low, svm_ci.high), (cnn_acc, cnn_ci.low, cnn_ci.high)]
    ):
        ax_acc.text(i, hi + 0.015, f"{acc:.3f}", ha="center")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_ylabel("Acurácia no teste")
    ax_acc.set_title("Acurácia com IC 95% (bootstrap)")

    ax_diff.hist(diff_dist, bins=40, color="#55a868", edgecolor="white")
    ax_diff.axvline(0, color="black", linestyle="--", label="diferença nula")
    ax_diff.axvline(diff_ci.low, color="red", linestyle=":", label="IC 95%")
    ax_diff.axvline(diff_ci.high, color="red", linestyle=":")
    ax_diff.axvline(diff_mean, color="#55a868", label=f"média = {diff_mean:+.3f}")
    ax_diff.set_xlabel("Diferença de acurácia (CNN - SVM)")
    ax_diff.set_ylabel("Frequência (bootstrap)")
    ax_diff.set_title("Distribuição bootstrap da diferença pareada")
    ax_diff.legend()

    fig.tight_layout()
    fig_path = RESULTS_DIR / "comparison_ci.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Gráfico salvo em {fig_path}")


if __name__ == "__main__":
    main()
