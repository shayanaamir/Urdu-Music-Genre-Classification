import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def extract_features(name, filepath=None):
    """
        The function extracts all the features of the specified file.

        Parameters:

            name: Name of the file the features should be extracted for
            filepath: Unless specified the default Filepath is the music folder in the same directory.

        Returns:
            Dictionary containing features and their values in a key:value format where key is the feature
            name and values in a value/numpy array.
    
    
    """
    data = {}
    if filepath!=None:
        filepath += "/" + name

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

    return data

def Generate_CSV(directory):
    """
    The function takes in the directory and traverses over it. It extracts all the features of the
    music file in the directory and returns a Result.csv file in the same directory.

    Parameters:
        directory: The folder directory containing wav files
        
    Returns:
        None.
        Returns a Result.csv file in the same directory.
    
    """
    # Fetching names of all files in directory
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

    # Creatig Empty DataFrame to store features of all files
    csv = pd.DataFrame(columns=['id', 'zcr', 'chroma_cqt', 'chroma_cens', 'tonnetz', 'chroma_stft',
       'rmse', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_rolloff', 'mfcc'])

    # Traversing over the file in the directory
    for i in onlyfiles:
        # Extracting featues of each music file
        data = pd.DataFrame(extract_features(i, filepath=directory), columns=csv.columns)

        # Appending data onto a dataframe
        csv = csv.append(data)

    csv.set_index("id", inplace=True)
    
    # Saving extracted data to a csv file
    csv.to_csv(directory+"/Result.csv", sep=",")

Generate_CSV("./music")