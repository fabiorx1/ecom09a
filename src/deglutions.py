import os, sys
from scipy.io import wavfile
from os.path import join, exists, split
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')
from src.inliers import INLIER_DIR
DEGLUTION_CHARTS = join('data', 'Charts-Deglution')

def fit():
    inlier_wav_files = [os.path.join(INLIER_DIR, f) for f in os.listdir(INLIER_DIR) if f.endswith('.wav')]
    X = [samples for _, samples in [wavfile.read(wav_file) for wav_file in inlier_wav_files]]
    s = []
    for x in X:
        dbscan = DBSCAN()
        dbscan.fit(x)
        y = dbscan.predict(x)
        s.append((x, y))
    return s

def plot_wav_charts_with_deglutions():
    wav_files = [os.path.join(INLIER_DIR, f) for f in os.listdir(INLIER_DIR) if f.endswith('.wav')]
    if not exists(DEGLUTION_CHARTS): os.makedirs(DEGLUTION_CHARTS)
    print(f'plotting {len(wav_files)} deglution wav charts...')
    s = fit()
    for k, wav_file in enumerate(wav_files):
        rate, data = wavfile.read(wav_file)
        _, y = s[k]
        plt.plot([i/rate for i in range(len(data))], data, c=y, cmap='tab10')
        plt.title(os.path.basename(wav_file))
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        fname = split(wav_file)[-1].split('.')[0]
        plt.savefig(join(DEGLUTION_CHARTS, fname))
        plt.clf()
    ...

if __name__ == '__main__':
    plot_wav_charts_with_deglutions()
    ...