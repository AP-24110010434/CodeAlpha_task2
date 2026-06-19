# Emotion Recognition from Speech

## Objective

The objective of this project is to recognize human emotions from speech audio using Machine Learning and speech signal processing techniques. The model analyzes audio recordings and predicts emotions such as Happy, Sad, Angry, Neutral, Fearful, Disgust, Calm, and Surprised.

---

## Approach

This project uses:

- Speech Signal Processing
- MFCC (Mel-Frequency Cepstral Coefficients) Feature Extraction
- Random Forest Classification

The audio files are processed to extract MFCC features, which are then used to train a machine learning model for emotion classification.

---

## Dataset

**Dataset Used:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

The dataset contains speech recordings from 24 actors expressing different emotions.

### Supported Emotions

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## Tools and Libraries Used

- Pandas
- NumPy
- Librosa
- Scikit-learn
- Joblib
- Soundfile

---

## Project Structure

```
Emotion_Recognition_Project/
│
├── Audio_Speech_Actors_01-24/
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ...
│
├── emotion_recognition.py
├── requirements.txt
└── README.md
```

---

## Steps Performed

1. Loaded audio files from the RAVDESS dataset.
2. Extracted MFCC features from each audio recording.
3. Mapped emotion labels using RAVDESS emotion codes.
4. Converted extracted features into numerical feature vectors.
5. Split the dataset into training and testing sets.
6. Trained a Random Forest Classifier.
7. Evaluated the model using classification metrics.
8. Saved the trained model for future use.

---

## Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## Sample Results

- Total Audio Samples: 1440
- Number of Emotion Classes: 8
- Model: Random Forest Classifier
- Accuracy: 60.42%

### Detected Emotions

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

### Output

```
Training Model...

Accuracy: 0.6042

Model saved successfully.
```

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Program

```bash
python emotion_recognition.py
```

---

## Outcome

The project successfully identifies emotions from speech recordings using MFCC feature extraction and a Random Forest classifier. The trained model achieves approximately 60% accuracy on the RAVDESS dataset and demonstrates the application of machine learning in speech emotion recognition.