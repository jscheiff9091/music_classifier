import numpy as np

def compute_mfccs(audio_sig, fs, fft_size, window_size=2048):
    # dtft_windows = fft_window(audio_sig, fs, fft_size, window_size)

    # ... fft_window()
    # loop
    # ... mfcc()
    # append to our matrix
    pass

def mfcc(fft_coefs, Nb=40):
    """
    Computes the mfcc coefficients for a 24s track snipit

    params
    ------
    fft_coefs : matrix {}
        fourier transform coefficeints of the track at T=24s
    Nb : int
        number of filters

    returns
    -------
    mfcc coefs : matrix { 40 x 258 }
        Mel Frequency Cepstral Coefficients
    """
    f_max = fs/2
    mel_min = np.log10(1 + 20/700)
    mel_max = np.log10(1 + f_max/700)
    mel_range = (mel_max - mel_min) / Nb


def fft_window(audio_sig, fs, fft_size, window_size=2048):
    """
    Computes the fourier transform of the signal in each window

    https://docs.scipy.org/doc/numpy/reference/routines.fft.html

    params
    ------
    audio_sig : 1D - vector
        audio signal
    fs : int
        sampling frequency
    fft_size : int
        Size of our fft?
    window_size : int
        Number of samples in each window
    num_frames : int
        Number of overlapping frames in the signal
        Denoted Nf in the document
    
    returns
    -------
    dtft_windows : matrix { }
    """
    

    # do something with numpy.fft( ... ) NICE it defaults to one-sided