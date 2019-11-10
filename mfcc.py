from __future__ import division
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

FILTER_BANK = 0
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

def mfcc(fft_coefs, fft_size, fs, Nb=40):
    """
    Computes the mfccs for a 24s track snipit

    params
    ------
    fft_coefs : matrix {}
        fourier transform coefficeints of the track up to T=24s
    fft_size : int
        Spacing in between linear frequencies from 0 to fs/2
    fs : int
        Rate of sampling of .wav file being analyzed
    Nb : int
        number of filters

    returns
    -------
    mfccs : matrix { 42 x 1 }  ???????
        Mel Frequency Cepstral Coefficients
    """

    #What we're gonna calculate
    mel_coef = np.zeros(42)

    for i in range(0,42):
        for j in range(0, 1024):
            mel_coef[i][j] = mel_coef[i][j] + (np.absolute(FILTER_BANK[i][j] * fft_coefs[j]))**2

    return mel_coef



def create_filter_bank(fs, Nb=40):
    """
    Creates filter bank of 40 (and two half) filters for computing MFCCs

    params
    ------
    fs : int
        Rate of sampling of .wav file being analyzed
    Nb : int
        number of filters

    returns
    -------
    filter_bank : matrix { 42 x 1024 }
        Mel Frequency Cepstral Coefficients
    """

    #constants
    f_max = fs/2
    lin_step = fs / 2048
    mel_const = 1127.01048

    #what we're gunna calculate :)
    FILTER_BANK = np.zeros((42, 1024))
    center_freq = np.zeros(42)
    lin_freq = np.zeros(1024)

    #Find Mel scale center frequencies   (mel = 1127.01048*log(1 + f/700))
    mel_min = mel_const * np.log10(1 + MIN_FREQ/700)
    mel_max = mel_const * np.log10(1 + f_max/700)
    mel_step = (mel_max - mel_min) / (Nb+1)

    #calculate linear scale center frequencies
    for i in range(0, 42):
        mel_equiv = mel_min + i * mel_step
        mel_equiv = 10**(mel_equiv / mel_const)
        center_freq[i] = (mel_equiv - 1) * 700

    #print(center_freq)

    #define filters between
    for i in range(0, 42):
        center = center_freq[i]
        if i == 0:
            left = 0
            right = center_freq[i+1]
            k = 1 / (right - center)
        elif i == 41:
            left = center_freq[i-1]
            right = 0
            k = 1 / (center - left)
        else:
            left = center_freq[i-1]
            right = center_freq[i+1]
            k = 2 / (right - left)

        for j in range(0, 1024):
            check_freq = lin_step*(j+1)
            if i == 0:
                lin_freq[j] = check_freq
                if check_freq < right and check_freq > center:
                    FILTER_BANK[i][j] = k * (right - check_freq) / (right - center)
                else:
                    FILTER_BANK[i][j] = 0
            elif i == 41:
                if check_freq < center and check_freq > left:
                    FILTER_BANK[i][j] = k * (check_freq - left) / (center - left)
                else:
                    FILTER_BANK[i][j] = 0
            else:
                if check_freq < center and check_freq > left:
                    FILTER_BANK[i][j] = k * (check_freq - left) / (center - left)
                elif check_freq < right and check_freq > center:
                    FILTER_BANK[i][j] = k * (right - check_freq) / (right - center)
                else:
                    FILTER_BANK[i][j] = 0
    
        #plt.plot(lin_freq, FILTER_BANK[i])
    #plt.xlabel("Frequency (Hz)")
    #plt.ylabel("Filter Magnitude")
    #plt.title("Cochlear Filter Bank")
    #plt.show()    



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