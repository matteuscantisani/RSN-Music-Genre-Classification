import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

WAV_DIR = Path("data_loader/data/wav")
SEG_DIR = Path("data_loader/data/segments")
SEGMENT_DURATION = 30
HOP_SECONDS = 5
SR = 22050

def compute_window_score(y_window, sr):
    rms = np.mean(librosa.feature.rms(y=y_window).ravel())

    contrast = librosa.feature.spectral_contrast(y=y_window, sr=sr)
    contrast_score = np.mean(contrast.ravel())

    centroid = librosa.feature.spectral_centroid(y=y_window, sr=sr)
    centroid_score = np.mean(centroid.ravel())

    mfcc = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=13)
    mfcc_var = np.mean(np.var(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y_window, sr=sr)
    chroma_score = np.mean(chroma.ravel())

    return {
        "rms": rms,
        "contrast": contrast_score,
        "centroid": centroid_score,
        "mfcc_var": mfcc_var,
        "chroma": chroma_score,
    }

def select_best_segment(y, sr):
    total_samples = len(y)
    segment_samples = SEGMENT_DURATION * sr
    hop_samples = HOP_SECONDS * sr

    if total_samples < segment_samples:
        return None

    scores = []
    for start in range(0, total_samples - segment_samples + 1, hop_samples):
        end = start + segment_samples
        window = y[start:end]
        window_scores = compute_window_score(window, sr)
        scores.append((start, window_scores))

    if not scores:
        return None

    metric_names = ["rms", "contrast", "centroid", "mfcc_var", "chroma"]
    for metric in metric_names:
        vals = np.array([s[1][metric] for s in scores])
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-12:
            for i in range(len(scores)):
                scores[i][1][metric] = (scores[i][1][metric] - vmin) / (vmax - vmin)
        else:
            for i in range(len(scores)):
                scores[i][1][metric] = 0.0

    best_start, best_metrics = max(
        scores, key=lambda s: np.mean([s[1][m] for m in metric_names])
    )

    return best_start, best_start + segment_samples

def process_all():
    genre_dirs = [d for d in sorted(WAV_DIR.iterdir()) if d.is_dir()]
    if not genre_dirs:
        raise FileNotFoundError("Nenhum diretório de gênero encontrado em data/wav/.")

    for genre_dir in genre_dirs:
        genre = genre_dir.name
        out_dir = SEG_DIR / genre
        out_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(f for f in genre_dir.iterdir() if f.suffix == ".wav")
        if not wav_files:
            continue

        for wav_path in wav_files:
            out_path = out_dir / f"{wav_path.stem}__best.wav"
            if out_path.exists():
                print(f"Já existe: {out_path}")
                continue

            y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
            duration = len(y) / sr

            if duration < SEGMENT_DURATION:
                print(
                    f" Pulando (menos de {SEGMENT_DURATION}s): "
                    f"{wav_path.name} ({duration:.1f}s)"
                )
                continue

            result = select_best_segment(y, sr)
            if result is None:
                print(f" Erro ao processar: {wav_path.name}")
                continue

            start_sample, end_sample = result
            segment = y[start_sample:end_sample]
            sf.write(str(out_path), segment, sr)
            start_sec = start_sample / sr
            end_sec = end_sample / sr
            print(
                f" Segmento salvo: {out_path.name} "
                f"[{start_sec:.1f}s - {end_sec:.1f}s]"
            )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    process_all()