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

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


