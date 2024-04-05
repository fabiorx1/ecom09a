import os, sys
from os.path import join, exists, split
import matplotlib.pyplot as plt
from scipy.io import wavfile

sys.path.append('.')
from src.convert import wav_dir

chart_dir = join('data', 'Charts-2')

if __name__ == '__main__':
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    if not exists(chart_dir): os.makedirs(chart_dir)
    for wav_file in wav_files:
        rate, data = wavfile.read(wav_file)
        plt.plot([i/rate for i in range(len(data))], data)
        plt.title(os.path.basename(wav_file))
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        fname = split(wav_file)[-1].split('.')[0]
        plt.savefig(join(chart_dir, fname))
        plt.clf()