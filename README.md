# Gender Classification from Speech using MFCCs

This project builds a complete machine learning pipeline to classify the gender of a speaker from short speech recordings. It is based on the extraction of Mel-Frequency Cepstral Coefficients (MFCCs), a widely used feature in speech signal processing, and demonstrates their practical connection to the Fourier Transform.

## Project Scope

Originally assigned to analyze vowel recordings using Fourier Transforms, this project extends the idea by:

* Using **windowed FFTs** to compute MFCCs from speech
* Applying **machine learning** to a real-world task
* Building a working classifier with over **98% accuracy**

## Dataset

* **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
* 1,440 audio-only speech samples from 24 professional actors (12 male, 12 female)

## Features

* **MFCCs (13)** + **delta** + **delta-delta** → 39 features per frame
* **Mean and standard deviation** summarization → 78-dim vector per sample

## Model

* **XGBoost classifier**
* Hyperparameter tuning via **GridSearchCV** with 5-fold cross-validation
* Final model achieves:

  * Accuracy: 98.6%
  * Precision: 99.3%
  * Recall: 97.9%
  * F1 Score: 98.6%

## Inference Pipeline

* A saved model (`.pkl`) and scaler are used to make predictions on new `.wav` files
* Inference script and notebook included to test any audio sample

## Files

* `ravdess_mfcc_pipeline.py`: Extracts MFCC features from RAVDESS
* `train_model.ipynb`: Training, evaluation, and model saving
* `inference_pipeline.py`: CLI-based prediction on new files
* `inference.ipynb`: Notebook for interactive inference

## How It Relates to Fourier Transform

MFCCs are computed using a series of Fourier Transforms (FFT) on windowed audio frames. These spectra are filtered using mel-scale triangular filters and compressed via DCT. This gives a perceptually meaningful, frequency-domain representation ideal for classification.

## Getting Started

1. Clone the repo
2. Install requirements (`librosa`, `joblib`, `xgboost`, `sklearn`, `numpy`)
3. Run the training notebook or inference notebook

---

This project demonstrates how core signal processing concepts like the Fourier Transform are applied in modern speech systems using MFCCs.
