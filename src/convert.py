import os
from pydub import AudioSegment
from os.path import join, exists

mp3_dir = join('data', 'MP3')
wav_dir = join('data', 'WAV')


def mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = os.path.split(mp3_file)[-1].split('.')[0] + '.wav'
    audio.export(join(wav_dir, wav_file), format='wav')
    return wav_file

if __name__ == '__main__':
    if not exists(wav_dir): os.makedirs(wav_dir)
    mp3_files = [os.path.join(mp3_dir, f) for f in os.listdir(mp3_dir) if f.endswith('.mp3')]
    wav_files = [mp3_to_wav(mp3_file) for mp3_file in mp3_files]