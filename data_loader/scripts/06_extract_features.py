import warnings
from pathlib import Path
import librosa
import numpy as np
import pandas as pd

# comentarios para melhor entendimento do codigo

SR = 22050 #taxa de amostragem
TOTAL_DURATION = 30
NUM_SEGMENTS = 10 
SAMPLES_PER_TRACK = SR * TOTAL_DURATION # nde pontos numericos 
SAMPLES_PER_SEGMENT = SAMPLES_PER_TRACK // NUM_SEGMENTS #total de amostras de um áudio

FINAL_DATASET_DIR = Path("data_loader/data/final_dataset")
FEATURES_DIR = Path("data_loader/data/features")

def extract_segment_features(y_seg, sr):
    features = {}

    # frequencia mel
    mfcc = librosa.feature.mfcc(y=y_seg, sr = sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfcc[i -1])
        features[f"mfcc{i}_var"] = np.var(mfcc[i - 1])

    #centroide espectral
    cs = librosa.feature.spectral_centroid(y=y_seg, sr=sr)
    features["spectral_centroid_mean"] = np.mean(cs)
    features["spectral_centroid_var"] = np.var(cs)

    #largura de banda espectral
    lb = librosa.feature.spectral_bandwidth(y=y_seg, sr=sr)
    features["spectral_bandwidth_mean"] = np.mean(lb)
    features["spectral_bandwidth_var"] = np.var(lb)

    # rolo espectral
    rolloff = librosa.feature.spectral_rolloff(y=y_seg, sr=sr)
    features["rolloff_mean"] = np.mean(rolloff)
    features["rolloff_var"] = np.var(rolloff)

    #cruzamentto por 0
    crz = librosa.feature.zero_crossing_rate(y=y_seg)
    features["zero_crossing_rate_mean"] = np.mean(crz)
    features["zero_crossing_rate_var"] = np.var(crz)

    #chroma stft
    chroma = librosa.feature.chroma_stft(y=y_seg, sr=sr)
    for i in range(1, 13):
        features[f"chroma{i}_mean"] = np.mean(chroma[i - 1])
        features[f"chroma{i}_var"] = np.var(chroma[i - 1])


    y_harm, y_perc = librosa.effects.hpss(y_seg)# separa harmonia e percussao

    #calcula o centroide tonal da harmoinia
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    for i in range(1, 7):
        features[f"tonnetz{i}_mean"] = np.mean(tonnetz[i - 1])
        features[f"tonnetz{i}_var"] = np.var(tonnetz[i - 1])

    # salva media(tlvz, mediana) e a variancia da harmonia e da percussao
    features["harmony_mean"] = np.mean(y_harm)
    features["harmony_var"] = np.var(y_harm)
    features["perceptr_mean"] = np.mean(y_perc)
    features["perceptr_var"] = np.var(y_perc)

    #calcula o rms
    rms = librosa.feature.rms(y=y_seg)
    features["rms_mean"] = np.mean(rms)
    features["rms_var"] = np.var(rms)

    #calcula o bpm
    tempo, _ = librosa.beat.beat_track(y=y_seg, sr=sr)
    features["tempo"] = float(np.atleast_1d(tempo)[0])

    return features

def extract_mel_spectrogram(y_seg, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(
        y=y_seg, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. Adiciona a dimensão de "canal" (Channel)
    # Bibliotecas como Keras/TensorFlow esperam tensores no formato (altura, largura, canais).
    # O espectrograma original tem o formato (n_mels, tempo).
    # O np.newaxis transforma isso em (n_mels, tempo, 1), simulando uma imagem de 1 canal.
    mel_spec_cnn = mel_spec_db[..., np.newaxis] 
    
    return mel_spec_cnn

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


    wav_files = sorted(FINAL_DATASET_DIR.glob("*.wav"))
    if not wav_files:
        print("Nenhum arquivo WAV encontrado em final_dataset/")
        return

    genres = sorted(set(f.name.split(".")[0] for f in wav_files))
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    n_tracks = len(wav_files)

    print(f"Gêneros: {genres}")
    print(f"Tracks: {n_tracks}")
    print(f"Segmentos por track: {NUM_SEGMENTS}")
    print()

    n_features = len(extract_segment_features(np.zeros(SAMPLES_PER_SEGMENT), SR))
    X_svm = np.zeros((n_tracks, n_features), dtype=np.float32)
    y_svm = np.zeros(n_tracks, dtype=np.int32)

    cnn_segments = []
    y_cnn_segments = []

    for idx, wav_path in enumerate(wav_files):
        genre = wav_path.name.split(".")[0]
        label = genre_to_idx[genre]
        y_svm[idx] = label

        if idx % 50 == 0:
            print(f"[{idx}/{n_tracks}] {wav_path.name}")

        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True, duration=TOTAL_DURATION)
        except Exception as e:
            print(f"Skipping {wav_path} because it is corrupted. Error: {e}")
            continue  # This will skip to the next audio file in your loop
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))

        seg_features = []
        for seg_idx in range(NUM_SEGMENTS):
            start = seg_idx * SAMPLES_PER_SEGMENT
            end = start + SAMPLES_PER_SEGMENT
            y_seg = y[start:end]

            seg_features.append(list(extract_segment_features(y_seg, sr).values()))

            mel = extract_mel_spectrogram(y_seg, sr)
            cnn_segments.append(mel)
            y_cnn_segments.append(label)

        X_svm[idx] = np.array(seg_features, dtype=np.float32).mean(axis=0)

    X_cnn = np.array(cnn_segments, dtype=np.float32)
    y_cnn = np.array(y_cnn_segments, dtype=np.int32)

    svm_dir = FEATURES_DIR / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    np.save(svm_dir / "X.npy", X_svm)
    np.save(svm_dir / "y.npy", y_svm)
    np.save(svm_dir / "classes.npy", np.array(genres))
    print(f"\nSVM: X {X_svm.shape}, y {y_svm.shape} -> {svm_dir}/")

    cnn_dir = FEATURES_DIR / "cnn"
    cnn_dir.mkdir(parents=True, exist_ok=True)
    print(f"Salvando {len(cnn_segments)} spectrograms CNN ...")
    np.save(cnn_dir / "X_segments.npy", X_cnn)
    np.save(cnn_dir / "y_segments.npy", y_cnn)
    np.save(cnn_dir / "classes.npy", np.array(genres))
    print(f"CNN: X {X_cnn.shape}, y {y_cnn.shape} -> {cnn_dir}/")

    print("\nExtracao concluida!")


if __name__ == "__main__":
    main()

