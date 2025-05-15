import os
import numpy as np
import librosa
import soundfile as sf
import argparse
from scipy.signal import butter, filtfilt

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def butter_bandpass(lowcut, highcut, sr, order=2):
    nyq = 0.5 * sr
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)
    if low >= high:
        raise ValueError(f"Invalid bandpass frequencies: low={lowcut}, high={highcut}, sr={sr}")
    return butter(order, [low, high], btype='band')

def bandpass_filter(signal, sr, lowcut=80.0, highcut=8000.0, order=2):
    b, a = butter_bandpass(lowcut, highcut, sr, order)
    return filtfilt(b, a, signal)

def spectral_subtraction(signal, sr, n_fft=2048, hop_length=512, noise_duration=0.5, oversub=1.5, floor_coef=0.1):
    num_noise = min(int(noise_duration * sr), signal.shape[0])
    if num_noise < 1:
        return signal
    noise_clip = signal[:num_noise]
    S_full = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    S_noise = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.mean(np.abs(S_noise), axis=1, keepdims=True)
    S_mag, S_phase = np.abs(S_full), np.angle(S_full)
    S_clean_mag = np.maximum(S_mag - oversub * noise_mag, floor_coef * noise_mag)
    S_clean = S_clean_mag * np.exp(1j * S_phase)
    return librosa.istft(S_clean, hop_length=hop_length, length=signal.shape[0])

def normalize_signal(signal):
    m = np.max(np.abs(signal))
    return signal / m if m > 0 else signal

def preprocess_audio(path, sr=16000):
    signal, _ = librosa.load(path, sr=sr)
    signal = pre_emphasis(signal)
    signal = bandpass_filter(signal, sr)
    signal = spectral_subtraction(signal, sr)
    signal = normalize_signal(signal)
    return signal, sr

def process_dataset(dataset_path, output_dir, sr=16000):
    for root, _, files in os.walk(dataset_path):
        rel = os.path.relpath(root, dataset_path)
        out_dir = os.path.join(output_dir, rel)
        os.makedirs(out_dir, exist_ok=True)
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue
            parts = fname.split('-')
            if len(parts) < 7 or parts[0] != '03' or parts[1] != '01':
                continue
            in_path = os.path.join(root, fname)
            sig, _ = preprocess_audio(in_path, sr)
            out_path = os.path.join(out_dir, fname)
            sf.write(out_path, sig, sr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sr', type=int, default=16000)
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output_dir, args.sr)
