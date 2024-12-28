import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import List, Tuple, Dict


def extract_features(audio_path: str) -> Dict[str, float]:
    """
    Extract audio features from a WAV file using librosa.

    Args:
        audio_path (str): Path to the WAV file

    Returns:
        Dict[str, float]: Dictionary of extracted features
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract features
    features = {}

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc{i + 1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i + 1}_std'] = np.std(mfccs[i])

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    return features


def load_dataset(positive_dir: str, negative_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load audio files from positive and negative directories and extract features.

    Args:
        positive_dir (str): Directory containing positive examples
        negative_dir (str): Directory containing negative examples

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and labels Series
    """
    features_list = []
    labels = []

    # Process positive examples
    for filename in os.listdir(positive_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(positive_dir, filename)
            try:
                features = extract_features(file_path)
                features_list.append(features)
                labels.append(1)  # Positive class
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Process negative examples
    for filename in os.listdir(negative_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(negative_dir, filename)
            try:
                features = extract_features(file_path)
                features_list.append(features)
                labels.append(0)  # Negative class
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Convert to DataFrame
    X = pd.DataFrame(features_list)
    y = pd.Series(labels)

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Train a Random Forest classifier on the audio features.

    Args:
        X (pd.DataFrame): Features DataFrame
        y (pd.Series): Labels Series
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_scaled)

    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Save the model and scaler
    joblib.dump(clf, 'audio_classifier.joblib')
    joblib.dump(scaler, 'audio_scaler.joblib')

    return clf, scaler

def train_model_from_dir(dir_pos, dir_neg):
    X, y = load_dataset(dir_pos, dir_neg)

    print(f"\nDataset summary:")
    print(f"Total samples: {len(y)}")
    print(f"Positive samples: {sum(y)}")
    print(f"Negative samples: {len(y) - sum(y)}")

    # Train the model
    print("\nTraining the model...")
    clf, scaler = train_model(X, y)

if __name__ == "__main__":
    train_model_from_dir("D:/repos/Data/PianoDataset05/Pos1", "D:/repos/Data/PianoDataset05/Neg1")