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
    """
    Demonstrates clipping if the tappered window is not used
    """
    print("Test clipping")
    last_time = 24
    window_size = 2048
    fft_size = 1000 #?

    # Load wavs
    fs, audio_sig = wavfile.read("wavs/chroma.wav")
    computed_audio = np.array(())

    for i in range( int(len(audio_sig) / window_size)):
        start_index = i*window_size
        end_index = (i+1)*window_size
        signal_window = audio_sig[start_index:end_index] # clipping here
        dtft = fft_window(signal_window).reshape((window_size,1))
        audio = np.fft.ifft(dtft)

        if computed_audio.size == 0:
            computed_audio = audio
        else:
            computed_audio = np.append(computed_audio, audio)

    scaled = np.int16(computed_audio.real/np.max(np.abs(computed_audio.real)) * 32767)
    write("clipping.wav", fs, scaled)

    # Verification of conversion of units 64 -> 16 bit functional
    # scaled2 = np.int16(audio_sig/np.max(np.abs(audio_sig)) * 32767)
    # write("normal.wav", fs, scaled2)
    
if __name__ == "__main__":
    # test_fft()
    test_clipping()
    # problem1()