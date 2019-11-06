from __future__ import division
from scipy.io import wavfile
from mfcc import *
from os import listdir

def problem1():
    print("Problem 1")
    last_time = 24

    audio_signals = {}

    # Load wavs
    for wav in listdir("wavs"):
        if "wav" in wav: # make sure it's a wav file
            fs, data = wavfile.read("wavs/" + wav)
            audio_signals[wav] = [fs, data]

    # Trim signals to 24s
    for wav in audio_signals:
        last_index = get_index_of_time(audio_signals[wav][0], last_time)
        audio_signals[wav][1] = audio_signals[wav][1][:last_index]
    
def get_index_of_time(fs, time):
    return int( fs * time )
    
if __name__ == "__main__":
    problem1()