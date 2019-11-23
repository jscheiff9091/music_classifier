import queue
import numpy as np
import matplotlib.pyplot as plt

def compute_max_frequencies(audio_sig, lin_frq, window_size=2048):
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
    fm_matrix : matrix {n x 515}
        semitones of the windowed fourier transform peak
    '''

    fm_matrix = np.array(())

    num_bins = int(2 * (audio_sig.size / window_size))
    for i in range(20):
        N = int(window_size/2)
        start_index = i* int(window_size/2)
        end_index = start_index + window_size

        #Get values of fft within window
        signal_window = np.float64(audio_sig[start_index:end_index].copy())
        
        #apply window
        signal_window *= np.hamming(window_size)

        dtft = abs(np.fft.fft(signal_window)) # absolute value to get mag of imag

        dtft = dtft.reshape((dtft.size, 1))

        #Find peaks
        smooth_fft = get_averaged_window(dtft[:1024])
        fm_vector, frq_peaks = find_peak_frequencies(smooth_fft[:1024], dtft[:1024], lin_frq)
        '''
        print(frq_peaks)
        plt.plot(lin_frq, (np.mean(dtft[:1024]) + np.std(dtft[:1024])) * np.ones(dtft[:1024].shape))
        plt.plot(lin_frq, smooth_fft[:1024])
        plt.plot(lin_frq, dtft[:1024])
        plt.show()
        '''

        #append to return matrix
        
        if i == 0:
            fm_matrix = fm_vector
        else:
            np.append(fm_matrix, fm_vector, axis=1)
    
    return fm_matrix

def get_averaged_window(window_fft):
    '''
    Reduce noise by averaging around each point in the FFT

    params
    -----
    window_fft : vector {1024 x 1}
        fft of windowed audio signal

    returns
    -------
    smooth_fft : vector {1024 x 1}
        averaged windowed audio signal
    '''

    #initialize variables
    #smooth_fft = np.array((window_fft.shape))
    smooth_fft = np.zeros(window_fft.shape)

    #find average of two adjacent frequencies and self
    for i in range(1, window_fft.size):
        if i == 0 or i == window_fft.size-1:
            pass
        elif i == 1:
            smooth_fft[i] = (window_fft[i-1] + window_fft[i] + window_fft[i+1]) / 3
            smooth_fft[i-1] = window_fft[i]
        elif i == window_fft.size-2:
            smooth_fft[i] = (window_fft[i-1] + window_fft[i] + window_fft[i+1]) / 3
            smooth_fft[i+1] = window_fft[i]
        else:
            smooth_fft[i] = (window_fft[i-1] + window_fft[i] + window_fft[i+1]) / 3
    
    return smooth_fft

def find_peak_frequencies(smooth_fft, window_fft, lin_frq):
    '''
    Compute vector containing magnitude squared of the fft at the peak frequencies

    params
    ------
    window_fft : vector {1024 x 1}
        fft of audio signal window

    returns
    -------
    fm_vector : {N x 1}
        semitones of the peak freuquencies
    '''

    #Create constants
    dev_weight = 1
    threshold = np.mean(window_fft) + dev_weight * np.std(window_fft)
    already_peak = False
    #frq_dict = {}            #delete
    #freq_0 = 27.5
    #freq_res = 11025 / 2048

    #Create Return Variable
    fm_vector = np.zeros(window_fft.shape)

    #loop through each element to and test if it is the peak value
    for i in range(window_fft.size):
        if i == 0:                        #Don't considder a frequency if it does not have two adjacent elements
            fm_vector[i] = 0
        elif i == window_fft.size-1:
            fm_vector[i] = 0
        else:
            #check if distinct peak
            if already_peak:
                #if new peak in region has greater magnitude than peak already found, set as new peak
                if window_fft[i] >= threshold and smooth_fft[i] >= threshold and window_fft[i] > window_fft[i-1] and window_fft[i] > window_fft[i+1]:
                    if window_fft[i] > window_fft[last_frq]:
                        fm_vector[last_frq] = 0
                        fm_vector[i] = (float(window_fft[i]))**2
                        #frq_dict[int(lin_frq[i])] = (int(window_fft[i]))**2
                        #frq_dict[int(lin_frq[last_frq])] = 0                                    #delete
                        last_frq = i
                    else:
                        fm_vector[i] = 0
                else:
                    fm_vector[i] = 0
                    
                #crossed back under threshold, start looking for new peak 
                if smooth_fft[i] < threshold:
                    already_peak = False
            else:
                if window_fft[i] >= threshold and smooth_fft[i] >= threshold and window_fft[i] > window_fft[i-1] and window_fft[i] > window_fft[i+1]:
                    fm_vector[i] = (float(window_fft[i]))**2
                    #frq_dict[int(lin_frq[i])] = (int(window_fft[i]))**2                          #delete
                    last_frq = i
                    already_peak = True
                else:
                    fm_vector[i] = 0
            

    return fm_vector, frq_dict



'''
Pretty sure we don't need this anymore (in this function)

frequency = i * freq_res
semitone = np.around(12 * np.log2(frequency / freq_0))
semitone = np.mod(semitone, 12)
if i == 0:
    sm_vector = semitone
    freq_vector = frequency
else:
    print(sm_vector)
    print(semitone)
    np.append(sm_vector, semitone, axis=0)
    np.append(freq_vector, frequency, axis=0)    #delete
'''