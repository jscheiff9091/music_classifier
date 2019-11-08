from __future__ import division
import numpy as np
from scipy.io.wavfile import write

MIN_FREQ = 20
# MAX_FREQ depends on each signal's fs = {fs/2}

def compute_mfccs(audio_sig, fs, fft_size, window_size=2048):
    # dtft_windows = fft_window(audio_sig, fs, fft_size, window_size)
    print(audio_sig.size)
    mfcc_array = np.array(())

    num_bins = int(2 * (audio_sig.size / window_size))
    for i in range(num_bins-1):
        start_index = i* int(window_size/2)
        end_index = start_index + window_size
        signal_window = np.float64(audio_sig[start_index:end_index].copy())
        
        signal_window *= np.hamming(window_size)

        dtft = np.fft.fft(signal_window)
        mfccs = mfcc(dtft, fft_size)

        if len(mfcc_array) == 0:
            mfcc_array = mfccs
        else:
            mfcc_array = np.append(mfcc_array, mfccs, axis=0) # May need to be axis 1 to work

def get_audio_window_tapered_hamming(audio_sig, start_index, end_index):
    # I think extract the window then do point wise multiplication with the hamming window
    pass

def mfcc(fft_coefs, fft_size, Nb=40):
    """
    Computes the mfccs for a 24s track snipit

    params
    ------
    fft_coefs : matrix {}
        fourier transform coefficeints of the track up to T=24s
    fft_size : int
        Spacing in between linear frequencies from 0 to fs/2
    Nb : int
        number of filters

    returns
    -------
    mfccs : matrix { 40 x 258 }
        Mel Frequency Cepstral Coefficients
    """
    hop_size = fft_size / 2
    f_max = fs/2
    mel_min = np.log10(1 + MIN_FREQ/700)
    mel_max = np.log10(1 + f_max/700)
    mel_range = (mel_max - mel_min) / Nb


def fft_window(audio_sig):
    """
    Computes the fourier transform of the signal

    https://docs.scipy.org/doc/numpy/reference/routines.fft.html

    params
    ------
    audio_sig : numpy array 1D
        audio signal
    
    returns
    -------
    dtft : array {1 x 2048}
        Symetric array around N/2
        let N = len(audio_sig)
        Indices: value
        0: 0 freq
        1: sample freq / N
        2: 2*sample_freq / N
        N/2: (repeat) 0 freq copy
        N/2+1: (repeat) sample freq / N
        ...

        to just access one half do dtft[:N//2] (assuming you have N, the signal window size)
    """
    return np.fft.fft(audio_sig)