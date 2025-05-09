import os
import numpy as np
import librosa
import argparse

def extract_mfcc_features(file_path, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta1, delta2])

def summarize_features(feature_matrix):
    means = feature_matrix.mean(axis=1)
    stds = feature_matrix.std(axis=1)
    return np.concatenate([means, stds])

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def parse_ravdess_filename(fname):
    parts = os.path.splitext(fname)[0].split('-')
    emotion = EMOTION_MAP.get(parts[2], 'unknown')
    actor_id = int(parts[6])
    gender = 0 if actor_id % 2 == 1 else 1
    return gender, emotion

def build_gender_dataset(dataset_path, out_features, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    X, y = [], []
    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue
            parts = os.path.splitext(fname)[0].split('-')
            if len(parts) < 7 or parts[0] != '03' or parts[1] != '01':
                continue
            gender, _ = parse_ravdess_filename(fname)
            path = os.path.join(root, fname)
            feats = extract_mfcc_features(path, sr, n_mfcc, n_fft, hop_length)
            vec = summarize_features(feats)
            X.append(vec)
            y.append(gender)
    X = np.vstack(X)
    y = np.array(y)
    np.savez(out_features, X=X, y=y)

def build_emotion_dataset(dataset_path, out_features, collapse_intensity=True, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    X, y = [], []
    if collapse_intensity:
        mapping = list(EMOTION_MAP.values())
    else:
        mapping = [f"{label}_{lvl}" for label in EMOTION_MAP.values() for lvl in (['normal'] if label=='neutral' else ['normal','strong'])]
    label_to_idx = {lbl: idx for idx, lbl in enumerate(mapping)}
    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue
            parts = os.path.splitext(fname)[0].split('-')
            if len(parts) < 7 or parts[0] != '03' or parts[1] != '01':
                continue
            _, emotion = parse_ravdess_filename(fname)
            lvl = 'normal' if parts[3] == '01' else 'strong'
            lbl = emotion if collapse_intensity else f"{emotion}_{lvl}"
            idx = label_to_idx.get(lbl)
            if idx is None:
                continue
            path = os.path.join(root, fname)
            feats = extract_mfcc_features(path, sr, n_mfcc, n_fft, hop_length)
            vec = summarize_features(feats)
            X.append(vec)
            y.append(idx)
    X = np.vstack(X)
    y = np.array(y)
    np.savez(out_features, X=X, y=y, mapping=mapping)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--out-features', required=True)
    parser.add_argument('--task', choices=['gender', 'emotion'], required=True)
    parser.add_argument('--no-collapse', action='store_true')
    args = parser.parse_args()
    if args.task == 'gender':
        build_gender_dataset(args.dataset_path, args.out_features)
    else:
        build_emotion_dataset(args.dataset_path, args.out_features, collapse_intensity=not args.no_collapse)
