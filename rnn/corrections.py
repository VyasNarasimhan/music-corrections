import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import sys

import copy

import re
from pyaudio import PyAudio, paInt16

model = tf.keras.models.load_model('models/kalyani.keras')

sample_rate = 48000
block_size = 1024

all_notes = ['E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#']
for i in range(len(all_notes)):
    all_notes.append(all_notes[i] + '_long')
    all_notes[i] += '_short'

streak = 0

threshold = 3

freq_log = []
log = []
cur_note = None

buffer = np.zeros(1024 * 4)
frames = 1024
padding = len(buffer) * 2
harmonics = 6    

def process_note(note):
    global cur_note, streak
    note = note.replace('♯', '#')
    i = re.search(r'\d', note)
    note = note[:i.start()]

    if not cur_note:
        cur_note = note
        streak = 1
    
    if note == cur_note:
        streak += 1
    else:
        if streak <= threshold:
            log.append(all_notes.index(f'{cur_note}_short'))
        else:
            log.append(all_notes.index(f'{cur_note}_long'))
        streak = 1
        cur_note = note
        if not valid():
            print('Wrong note')

def valid():
    if len(log) < 6:
        return True
    sequence = log[-6:-1]
    y = log[-1]
    res = model.predict(np.array([sequence]))[0]
    sorted_res = np.sort(res)[::-1]
    print(np.where(sorted_res==res[y])[0][0])
    # print(res[y])
    return np.where(sorted_res==res[y])[0][0] < 5
    # return np.argmax(res) == y


def hps(samples, samplerate): # https://github.com/TomSchimansky/GuitarTuner/blob/master/tuner_audio/audio_analyzer.py

    buffer[:-frames] = buffer[frames:]
    buffer[-frames:] = samples

    # applying hanning window to reduce spectral leakage
    window = np.array(buffer) * np.hanning(len(buffer))

    # zero padding
    amplitudes = abs(np.fft.fft(np.pad(window, (0, padding))))
    # only use the first half of the fft output data
    amplitudes = amplitudes[:int(len(amplitudes) / 2)]

    frequencies = np.fft.fftfreq(len(amplitudes) * 2, 1 / samplerate)

    # HPS: multiply data by itself with different scalings (Harmonic Product Spectrum)
    hps_spectrum = copy.deepcopy(amplitudes)
    for i in range(2, harmonics + 1):
        multiples = amplitudes[::i] 
        hps_spectrum[:len(multiples)] *= amplitudes[::i]

    return frequencies[np.argmax(hps_spectrum)]

# Start real-time audio input
audio_object = PyAudio()
stream = audio_object.open(format=paInt16, channels=1, rate=sample_rate, input=True, output=False, frames_per_buffer=frames)

while True:
    try:
        data = stream.read(frames, exception_on_overflow=False)
        data = np.frombuffer(data, dtype=np.int16)
        freq = hps(data, sample_rate)
        if freq > 130:
            note = librosa.hz_to_note(freq)
            print(note)
            freq_log.append(freq)
            process_note(note)
    except KeyboardInterrupt:
        print(freq_log)
        sys.exit()
    except Exception as exc:
        print(str(exc))
    