import librosa
import numpy as np
import os

def extract_features(file_path, max_len=50):  # max_len: Maximum sequence length
    y, sr = librosa.load(file_path, sr=16000)  # Load audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCCs
    mfccs = mfccs.T  # Transpose to (time_steps, features)

    # Pad or truncate to fixed length
    if mfccs.shape[0] < max_len:
        pad_width = max_len - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')  # Pad
    else:
        mfccs = mfccs[:max_len, :]  # Truncate

    return mfccs

def load_data(folder_path, max_len=50):
    X, y = [], []
    labels = os.listdir(folder_path)
    label_dict = {label: idx for idx, label in enumerate(labels)}  # Map labels to numbers

    for label in labels:
        label_path = os.path.join(folder_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            features = extract_features(file_path, max_len)  # Extract padded/truncated features
            X.append(features)
            y.append(label_dict[label])  # Convert label to number

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
