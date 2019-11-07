from __future__ import division
from scipy.io import wavfile
from mfcc import *
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

def problem1():
    print("Problem 1")
    last_time = 24
    window_size = 2048
    fft_size = 1000 #?

    audio_signals = {}

    # Load wavs
    for wav in listdir("wavs"):
        if wav == "chroma.wav": # ignore chroma
            pass
        elif "wav" in wav: # make sure it's a wav file
            fs, data = wavfile.read("wavs/" + wav)
            audio_signals[wav] = [fs, data]

    # Trim signals to 24s
    for wav in audio_signals:
        last_index = get_index_of_time(audio_signals[wav][0], last_time)
        audio_signals[wav][1] = audio_signals[wav][1][:last_index]

    # Calculate mfcc's
    for wav in audio_signals:
        print("MFCC's : " + wav)
        fs, data = audio_signals[wav]
        compute_mfccs(data, fs, fft_size, window_size)
    
def get_index_of_time(fs, time):
    return int( fs * time )

def test_fft():
    t = np.linspace(0, 0.5, 500)
    s = np.sin(40 * 2 * np.pi * t)
    # plt.ylabel("Amp")
    # plt.xlabel("Time [s")
    # plt.plot(t,s)
    # plt.show()
    fft = np.fft.fft(s)
    T = t[1] - t[0] # sample period should be 0.1
    print(T)
    N = s.size

    f = np.linspace(0, 1/T, N)

    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1/N, width=1.5)  # 1 / N is a normalization factor
    plt.show()

    # I should do the dtft and then put in time to see if I hear clipping

def test_clipping():
    print("Test clipping")
    last_time = 24
    window_size = 2048
    fft_size = 1000 #?

    audio_signals = {}

    # Load wavs
    fs, data = wavfile.read("wavs/chroma.wav")
    compute_mfccs(data, fs, fft_size, window_size, tapered=False)

    # load one of the songs, perform fft in time, perform ifft and play back -> clipping?
    # then test if we apply the hamming window


    
if __name__ == "__main__":
    # test_fft()
    test_clipping()
    # problem1()