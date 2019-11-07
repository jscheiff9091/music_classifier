from __future__ import division
import numpy as np
from scipy.io.wavfile import write

MIN_FREQ = 20
# MAX_FREQ depends on each signal's fs = {fs/2}

def compute_mfccs(audio_sig, fs, fft_size, window_size=2048, tapered=True):
    # dtft_windows = fft_window(audio_sig, fs, fft_size, window_size)
    print(audio_sig.size)
    dtft_windows = np.array(())
    computed_audio = np.array(())

    for i in range( int(len(audio_sig) / window_size)):
        start_index = i*window_size
        end_index = (i+1)*window_size
        if tapered:
            pass
        else:
            signal_window = get_windowed_audio_no_taper(audio_sig, start_index, end_index)
        dtft = fft_window(signal_window).reshape((window_size,1))
        audio = np.fft.ifft(dtft)
        # break # TODO append to an array and play back # TODO call mfccs()

        if dtft_windows.size == 0:
            dtft_windows = dtft
            computed_audio = audio
        else:
            dtft_windows = np.append(dtft_windows, dtft, axis=1)
            computed_audio = np.append(computed_audio, audio)

    print(dtft_windows.shape)
    scaled = np.int16(computed_audio.real/np.max(np.abs(computed_audio.real)) * 32767)
    write("test.wav", fs, scaled)
    scaled2 = np.int16(audio_sig/np.max(np.abs(audio_sig)) * 32767)
    write("test2.wav", fs, scaled2)

# data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
# scaled = np.int16(data/np.max(np.abs(data)) * 32767)
# write('test.wav', 44100, scaled)
        

def get_windowed_audio_no_taper(audio_sig, start_index, end_index):
    return audio_sig[start_index:end_index]

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