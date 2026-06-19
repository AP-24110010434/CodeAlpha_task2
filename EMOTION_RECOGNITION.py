import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from joblib import dump

# ======================================
# FEATURE EXTRACTION
# ======================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)


# ======================================
# EMOTION MAPPING
# ======================================

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ======================================
# DATASET PATH
# ======================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Audio_Speech_Actors_01-24"
)

print("Dataset Path:")
print(DATASET_PATH)

# ======================================
# LOAD DATASET
# ======================================

features = []
labels = []

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(
        DATASET_PATH,
        actor
    )

    if not os.path.isdir(actor_path):
        continue

    print("Reading", actor)

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        try:

            parts = file.split("-")

            emotion_code = parts[2]

            if emotion_code not in emotion_map:
                continue

            emotion = emotion_map[
                emotion_code
            ]

            file_path = os.path.join(
                actor_path,
                file
            )

            feature = extract_features(
                file_path
            )

            features.append(feature)
            labels.append(emotion)

        except Exception as e:

            print(
                f"Error processing {file}: {e}"
            )

# ======================================
# CONVERT TO ARRAYS
# ======================================

X = np.array(features)
y = np.array(labels)

print("\nTotal Samples:", len(X))
print("Emotions:", np.unique(y))

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================
# RANDOM FOREST MODEL
# ======================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

# ======================================
# PREDICTIONS
# ======================================

y_pred = model.predict(
    X_test
)

# ======================================
# EVALUATION
# ======================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "="*50)
print("EMOTION RECOGNITION RESULTS")
print("="*50)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ======================================
# SAVE MODEL
# ======================================

models_dir = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    models_dir,
    exist_ok=True
)

dump(
    model,
    os.path.join(
        models_dir,
        "emotion_model.pkl"
    )
)

print("\nModel saved successfully.")