import os, sys, shutil
from scipy.io import wavfile
from os.path import join, exists, split

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt


sys.path.append('.')
from src.convert import wav_dir


def compute_statistics(samples):
    mean = np.mean(samples)
    std_dev = np.std(samples)
    min_val = np.min(samples)
    max_val = np.max(samples)
    median = np.median(samples)
    q1 = np.percentile(samples, 25)
    q3 = np.percentile(samples, 75)
    
    return {
        'Mean': mean,
        'Standard Deviation': std_dev,
        'Minimum': min_val,
        'Maximum': max_val,
        'Median': median,
        'Q1': q1,
        'Q3': q3
        }

def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter([k[0] for k in X_pca],
                [k[1] for k in X_pca],
                c=y, cmap='viridis')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('2D PCA Plot with Outliers')
    plt.colorbar(label='Anomaly Score'), plt.grid(True)
    plt.savefig(join('data', 'OUTLIERS_PCA'))

def fit():
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    wav_samples = [samples for _, samples in [wavfile.read(wav_file) for wav_file in wav_files]]
    X = [list(compute_statistics(x).values()) for x in wav_samples]
    clf = IsolationForest(n_estimators=48)
    print('fitting forest...')
    clf.fit(X)
    print('applying forest...')
    y = clf.predict(X)
    score = clf.score_samples(X)
    return X, y, score

INLIER_DIR = join('data', 'WAV', 'inliers')
OUTLIER_DIR = join('data', 'WAV', 'outliers')
CLASSIFIED_CHART_DIR = join('data', 'Charts-Classified')

def save_classified_wav_files(y):
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    if not exists(INLIER_DIR): os.makedirs(INLIER_DIR)
    if not exists(OUTLIER_DIR): os.makedirs(OUTLIER_DIR)
    print(f'saving {len(wav_files)} classified wav files...')
    for i, wav_file in enumerate(wav_files):
        fname = split(wav_file)[-1]
        if y[i] > 0: shutil.copyfile(wav_file, join(INLIER_DIR, fname))
        else: shutil.copyfile(wav_file, join(OUTLIER_DIR, fname))

def plot_classified_wav_charts():
    inlier_wav_files = [os.path.join(INLIER_DIR, f) for f in os.listdir(INLIER_DIR) if f.endswith('.wav')]
    outlier_wav_files = [os.path.join(OUTLIER_DIR, f) for f in os.listdir(OUTLIER_DIR) if f.endswith('.wav')]
    wav_files = [(True, wav_file) for wav_file in inlier_wav_files]
    wav_files += [(False, wav_file) for wav_file in outlier_wav_files]
    if not exists(CLASSIFIED_CHART_DIR): os.makedirs(CLASSIFIED_CHART_DIR)
    print(f'plotting {len(wav_files)} classified wav charts...')
    for is_inlier, wav_file in wav_files:
        rate, data = wavfile.read(wav_file)
        plt.plot([i/rate for i in range(len(data))], data, 'b' if is_inlier else 'r')
        plt.title(os.path.basename(wav_file))
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        fname = split(wav_file)[-1].split('.')[0]
        plt.savefig(join(CLASSIFIED_CHART_DIR, fname))
        plt.clf()
    ...

if __name__ == '__main__':
    X, y, score = fit()
    plot_pca(X, score)
    save_classified_wav_files(y)
    plot_classified_wav_charts()
    ...