import os
import warnings
from pathlib import Path
import librosa
import numpy as np
from numpy.lib.format import open_memmap

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

    # descobre as dimensoes das features usando um segmento de silencio
    n_features = len(extract_segment_features(np.zeros(SAMPLES_PER_SEGMENT), SR))
    mel_shape = extract_mel_spectrogram(np.zeros(SAMPLES_PER_SEGMENT), SR).shape

    svm_dir = FEATURES_DIR / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    cnn_dir = FEATURES_DIR / "cnn"
    cnn_dir.mkdir(parents=True, exist_ok=True)

    # Os espectrogramas sao escritos direto no disco (memmap) em vez de
    # acumulados em RAM: o dataset completo tem ~1 GB e nao cabe na memoria
    # junto com o resto do processo.
    tmp_cnn_path = cnn_dir / "X_segments.tmp.npy"
    max_segments = n_tracks * NUM_SEGMENTS
    X_cnn = open_memmap(
        tmp_cnn_path, mode="w+", dtype=np.float32, shape=(max_segments, *mel_shape)
    )

    # listas em vez de arrays pre-alocados: arquivos corrompidos sao pulados
    # sem deixar linhas zeradas com label valido no dataset do SVM
    svm_rows = []
    svm_labels = []
    y_cnn = []
    track_ids = []

    n_valid_tracks = 0
    n_segments_written = 0
    n_skipped = 0

    for idx, wav_path in enumerate(wav_files):
        genre = wav_path.name.split(".")[0]
        label = genre_to_idx[genre]

        if idx % 50 == 0:
            print(f"[{idx}/{n_tracks}] {wav_path.name}")

        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True, duration=TOTAL_DURATION)
        except Exception as e:
            print(f"Pulando {wav_path.name} (corrompido): {e}")
            n_skipped += 1
            continue
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))

        # processa a track inteira antes de gravar qualquer coisa, para que
        # uma falha no meio nao deixe dados parciais nos arquivos de saida
        seg_features = []
        track_mels = []
        for seg_idx in range(NUM_SEGMENTS):
            start = seg_idx * SAMPLES_PER_SEGMENT
            end = start + SAMPLES_PER_SEGMENT
            y_seg = y[start:end]

            seg_features.append(list(extract_segment_features(y_seg, sr).values()))
            track_mels.append(extract_mel_spectrogram(y_seg, sr))

        for mel in track_mels:
            X_cnn[n_segments_written] = mel
            y_cnn.append(label)
            track_ids.append(n_valid_tracks)
            n_segments_written += 1

        svm_rows.append(np.array(seg_features, dtype=np.float32).mean(axis=0))
        svm_labels.append(label)
        n_valid_tracks += 1

    X_cnn.flush()
    del X_cnn

    if n_valid_tracks == 0:
        print("Nenhuma track valida processada.")
        tmp_cnn_path.unlink()
        return

    if n_skipped:
        print(f"\n{n_skipped} arquivo(s) corrompido(s) ignorado(s)")

    X_svm = np.stack(svm_rows).astype(np.float32)
    y_svm = np.array(svm_labels, dtype=np.int32)
    assert X_svm.shape == (n_valid_tracks, n_features)

    np.save(svm_dir / "X.npy", X_svm)
    np.save(svm_dir / "y.npy", y_svm)
    np.save(svm_dir / "classes.npy", np.array(genres))
    print(f"\nSVM: X {X_svm.shape}, y {y_svm.shape} -> {svm_dir}/")

    # finaliza o X do CNN: se nenhuma track foi pulada o arquivo temporario ja
    # esta completo; senao, copia so os segmentos validos (em blocos, sem
    # carregar tudo na RAM) e descarta o resto
    final_cnn_path = cnn_dir / "X_segments.npy"
    print(f"Salvando {n_segments_written} espectrogramas CNN ...")
    if n_segments_written == max_segments:
        os.replace(tmp_cnn_path, final_cnn_path)
    else:
        src = np.load(tmp_cnn_path, mmap_mode="r")
        dst = open_memmap(
            final_cnn_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_segments_written, *mel_shape),
        )
        chunk = 500
        for start in range(0, n_segments_written, chunk):
            end = min(start + chunk, n_segments_written)
            dst[start:end] = src[start:end]
        dst.flush()
        del src, dst
        tmp_cnn_path.unlink()

    y_cnn = np.array(y_cnn, dtype=np.int32)
    track_ids = np.array(track_ids, dtype=np.int32)
    np.save(cnn_dir / "y_segments.npy", y_cnn)
    np.save(cnn_dir / "track_ids.npy", track_ids)
    np.save(cnn_dir / "classes.npy", np.array(genres))
    print(f"CNN: X ({n_segments_written}, {mel_shape[0]}, {mel_shape[1]}, {mel_shape[2]}), "
          f"y {y_cnn.shape}, track_ids {track_ids.shape} -> {cnn_dir}/")

    print("\nExtracao concluida!")


if __name__ == "__main__":
    main()

