from __future__ import division
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

MIN_FREQ = 20
# MAX_FREQ depends on each signal's fs = {fs/2}

def compute_mfccs(filter_bank, audio_sig, window_size=2048):
    """
    Retuns the mfccs of the audio signal with the given signal size

    params
    ------
    filter_bank : matrix {40 x 1024}
        mfcc filter banks
    audio_sig : vector {529,200 x 1}
        the audio signal
    window_size : int
        window size of the signal to use

    returns
    -------
    mfcc_array : matrix {40 x 515}
        the mfcc array
    """
    # dtft_windows = fft_window(audio_sig, fs, fft_size, window_size)
    print(audio_sig.size)
    mfcc_array = np.array(())

    audio_array = np.array(())
    num_bins = int(2 * (audio_sig.size / window_size))
    for i in range(num_bins-1):
        N = int(window_size/2)
        start_index = i* int(window_size/2)
        end_index = start_index + window_size
        signal_window = np.float64(audio_sig[start_index:end_index].copy())
        
        signal_window *= np.hamming(window_size)

        # audio = np.fft.ifft(np.fft.fft(signal_window))
        # audio = audio.reshape((audio.size, 1))
        dtft = abs(np.fft.fft(signal_window)) # absolute value to get mag of imag
        # print(dtft.shape)

        dtft = dtft.reshape((dtft.size, 1))
        # print(abs(dtft[200:220]))
        # print(dtft.shape)
        mfccs = mfcc(filter_bank, dtft) 
        # print(mfccs.shape)

        if len(mfcc_array) == 0:
            # audio_array = audio
            mfcc_array = mfccs
        else:
            # print("looping")
            # audio_array[-N:] = audio_array[-N:] + audio[:N]
            # audio_array = np.append(audio_array, audio[N:])
            # audio_array = audio_array.reshape((audio_array.size,1))
            mfcc_array = np.append(mfcc_array, mfccs, axis=1) # May need to be axis 1 to work
    # scaled = np.int16(audio_array.real/np.max(np.abs(audio_array.real)) * 32767)
    # write("audio_hamming.wav", fs, scaled)
    return mfcc_array


def get_audio_window_tapered_hamming(audio_sig, start_index, end_index):
    # I think extract the window then do point wise multiplication with the hamming window
    pass

def mfcc(filter_bank, fft_vector):
    """
    Computes the mfccs for a 24s track snipit

    params
    ------
    filter_bank : matrix {40 x 1024}
        audio filter bank
    fft_vector : vector {2048 x 1}
        fourier transform coefficeints of the track up to T=24s
    returns
    -------
    mfccs : matrix { 40 x 1 }
        Mel Frequency Cepstral Coefficients
    """

    N = fft_vector.size # 2048
    one_sided_fft = fft_vector[:N//2]
    one_sided_fft_squared = np.square(one_sided_fft)
    filter_bank_squared = np.square(filter_bank)

    return np.dot(filter_bank_squared, one_sided_fft_squared)

    # for i in range(0,42):
        # for j in range(0, 1024):
            # mel_coef[i][j] = mel_coef[i][j] + (np.absolute(filter_bank[i][j] * fft_coefs[j]))**2
    # return mel_coef



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
    filter_bank : matrix { 40 x 1024 }
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

    
    return FILTER_BANK[1:-1,:]
    
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