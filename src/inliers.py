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
    clf = IsolationForest(n_estimators=16)
    clf.fit(X)
    y = clf.predict(X)
    y2 = clf.score_samples(X)
    return X, y, y2

inlier_dir = join('data', 'WAV', 'inliers')
outlier_dir = join('data', 'WAV', 'outliers')

def save(y):
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    if not exists(inlier_dir): os.makedirs(inlier_dir)
    if not exists(outlier_dir): os.makedirs(outlier_dir)
    for i, wav_file in enumerate(wav_files):
        fname = split(wav_file)[-1]
        if y[i] > 0: shutil.copyfile(wav_file, join(inlier_dir, fname))
        else: shutil.copyfile(wav_file, join(outlier_dir, fname))

CLASSIFIED_CHART_DIR = join('data', 'Charts-Classified')

def plot_samples():
    wav_files = [(True, os.path.join(inlier_dir, f)) for f in os.listdir(inlier_dir) if f.endswith('.wav')]
    wav_files += [(False, os.path.join(outlier_dir, f)) for f in os.listdir(outlier_dir) if f.endswith('.wav')]
    if not exists(CLASSIFIED_CHART_DIR): os.makedirs(CLASSIFIED_CHART_DIR)
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
    # X, y, y2 = fit()
    # plot_pca(X, y2)
    # save(y)
    plot_samples()
    ...