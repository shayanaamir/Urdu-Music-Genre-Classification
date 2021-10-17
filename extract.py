import librosa
import numpy as np
import pandas as pd

def extract_features(name, filepath="./music/"):
    data = {}
    if filepath!=None:
        filepath += name

    x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

    data['id'] = name

    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    data['zcr'] = [f]

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                n_bins=7*12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    data['chroma_cqt'] = [f]
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    data['chroma_cens'] = [f]
    f = librosa.feature.tonnetz(chroma=f)
    data['tonnetz'] = [f]

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
    del x

    f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    data['chroma_stft'] = [f]

    f = librosa.feature.rms(S=stft)
    data['rmse'] = [f]

    f = librosa.feature.spectral_centroid(S=stft)
    data['spectral_centroid'] = [f]
    f = librosa.feature.spectral_bandwidth(S=stft)
    data['spectral_bandwidth'] = [f]
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    data['spectral_contrast'] = [f]
    f = librosa.feature.spectral_rolloff(S=stft)
    data['spectral_rolloff'] = [f]

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    data['mfcc'] = [f]

    features = pd.DataFrame(data)
    return features

d = extract_features("The Kid LAROI, Justin Bieber - STAY (Official Video).wav")
print(d)