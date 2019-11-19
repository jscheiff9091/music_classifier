import queue

def compute_pitch(audio_sig, window_size=2048):
    '''
    Computes matrix with peak values of the windowed fourier transforms of the audio signal.

    params
    ------
    audio_sig : vector {529,200 x 1}
        the audio signal
    window_size : int
        window size of the signal to use

    returns
    -------
    sm_array : matrix {n x 515}
        semitones of the windowed fourier transform peak
    '''

    sm_array = np.array(())

    num_bins = int(2 * (audio_sig.size / window_size))
    for i in range(num_bins-1):
        N = int(window_size/2)
        start_index = i* int(window_size/2)
        end_index = start_index + window_size
        signal_window = np.float64(audio_sig[start_index:end_index].copy())
        
        signal_window *= np.hamming(window_size)

        
        dtft = abs(np.fft.fft(signal_window)) # absolute value to get mag of imag

        dtft = dtft.reshape((dtft.size, 1))

        #Find peak
        #append to semitone array

def find_peak_semitones(window_fft):
    '''
    Compute vector of peak semitones for an fft

    params
    ------
    window_fft : vector {1024 x 1}
        fft of audio signal window

    returns
    -------
    sm_vector : {N x 1}
        semitones of the peak freuquencies
    '''

    #Create threshhold

    #loop through each element to and test if it is the peak value