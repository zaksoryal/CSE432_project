"""
Feature extraction module for RAVDESS Speech Emotion Recognition.

Extracts a 192-dimensional fixed-length feature vector per audio file:
  MFCCs (13 × 4 stats)       = 52
  MFCC deltas (13 × 4 stats) = 52
  Chroma (12 × 4 stats)      = 48
  Mel spectrogram (20 mean)  = 20
  ZCR (4 stats)              =  4
  RMS energy (4 stats)       =  4
  Spectral centroid (4 stats)=  4
  Spectral bandwidth (4 stats)=  4
  Spectral rolloff (4 stats) =  4
                          Total = 192

Stats order for multi-dim features: [mean_0..mean_n, std_0..std_n, min_0..min_n, max_0..max_n]
"""

import pathlib
import numpy as np
import pandas as pd
import librosa

# Constants
SR         = 22050   # target sample rate in Hz; librosa resamples every file to this
N_MFCC     = 13      # number of MFCC coefficients to compute
N_MELS     = 20      # number of mel frequency bands for the mel spectrogram
HOP_LENGTH = 512     # samples between successive frames (~23ms at 22050Hz)
N_FFT      = 2048    # FFT window size in samples (~93ms at 22050Hz)
N_FEATURES = 192     # expected total length of the output feature vector


# Helper
def _multidim_stats(x: np.ndarray) -> np.ndarray:
    """
    Takes a 2D array of shape (n_features, n_frames) and returns a 1D array of length n_features*4
    containing the mean, std, min, and max of each feature across frames
    """
    return np.concatenate([
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
    ])


def _scalar_stats(x: np.ndarray) -> np.ndarray:
    """""Similar to Helper but for 1D arrays, returns mean, std, min, max"""
    return np.array([x.mean(), x.std(), x.min(), x.max()])


# Feature Extraction
def extract_features(filepath: str | pathlib.Path, sr: int = SR) -> np.ndarray:
    """
    Load one WAV file and return a 192-d feature vector
    """
    y, _ = librosa.load(filepath, sr=sr, mono=True)

    # 1. MFCCs  (13 × 4 = 52)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_feat = _multidim_stats(mfcc)

    # 2. MFCC deltas  (13 × 4 = 52)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_feat = _multidim_stats(mfcc_delta)

    # 3. Chroma STFT  (12 × 4 = 48)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma_feat = _multidim_stats(chroma)

    # 4. Mel spectrogram mean per band  (20)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_feat = mel_db.mean(axis=1)          # (N_MELS,)

    # 5. Zero Crossing Rate  (4)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    zcr_feat = _scalar_stats(zcr)

    # 6. RMS Energy  (4)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_feat = _scalar_stats(rms)

    # 7. Spectral Centroid  (4)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                 n_fft=N_FFT,
                                                 hop_length=HOP_LENGTH)[0]
    centroid_feat = _scalar_stats(centroid)

    # 8. Spectral Bandwidth  (4)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                   n_fft=N_FFT,
                                                   hop_length=HOP_LENGTH)[0]
    bandwidth_feat = _scalar_stats(bandwidth)

    # 9. Spectral Rolloff  (4)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                               n_fft=N_FFT,
                                               hop_length=HOP_LENGTH)[0]
    rolloff_feat = _scalar_stats(rolloff)

    feat = np.concatenate([
        mfcc_feat, mfcc_delta_feat, chroma_feat, mel_feat,
        zcr_feat, rms_feat, centroid_feat, bandwidth_feat, rolloff_feat,
    ])
    assert feat.shape == (N_FEATURES,), f"Expected {N_FEATURES} features, got {feat.shape[0]}"
    return feat


# Feature Names
def get_feature_names() -> list[str]:
    """Return list of 192 feature name strings matching extract_features output order."""
    names = []
    stats = ['mean', 'std', 'min', 'max']

    for stat in stats:
        for i in range(N_MFCC):
            names.append(f'mfcc_{i+1}_{stat}')

    for stat in stats:
        for i in range(N_MFCC):
            names.append(f'mfcc_delta_{i+1}_{stat}')

    for stat in stats:
        for i in range(12):
            names.append(f'chroma_{i+1}_{stat}')

    for i in range(N_MELS):
        names.append(f'mel_{i+1}_mean')

    for stat in stats:
        names.append(f'zcr_{stat}')

    for stat in stats:
        names.append(f'rms_{stat}')

    for stat in stats:
        names.append(f'centroid_{stat}')

    for stat in stats:
        names.append(f'bandwidth_{stat}')

    for stat in stats:
        names.append(f'rolloff_{stat}')

    assert len(names) == N_FEATURES, f"Expected {N_FEATURES} names, got {len(names)}"
    return names


# ─── Batch Extraction ────────────────────────────────────────────────────────
def extract_all(metadata: pd.DataFrame,
                sr: int = SR,
                verbose: bool = True) -> pd.DataFrame:
    """
    Run extract_features on every row in metadata.csv and return a DataFrame
    with one row per file and columns = get_feature_names().
    """
    from tqdm import tqdm

    feature_names = get_feature_names()
    rows = []
    filepaths = metadata['filepath'].tolist()

    iterable = tqdm(filepaths, desc='Extracting features', unit='file') if verbose else filepaths
    for fp in iterable:
        rows.append(extract_features(fp, sr=sr))

    return pd.DataFrame(rows, columns=feature_names, index=metadata.index)
