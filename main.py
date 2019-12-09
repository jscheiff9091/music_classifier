from __future__ import division
from scipy.io import wavfile
from mfcc import *
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pitch import *
from klg import *

def problem1():
    print("Problem 1")
    last_time = 24
    window_size = 2048

    audio_dict = {}

    # Load wavs
    for wav in listdir("wavs"):
        if wav == "chroma.wav": # ignore chroma
            pass
        elif "wav" in wav: # make sure it's a wav file
            fs, data = wavfile.read("wavs/" + wav)
            audio_dict[wav] = [fs*2, data] # wavfile.read returns fs/2 (for some reason...)

    # Trim signals to 24s
    for wav in audio_dict:
        last_index = get_index_of_time(audio_dict[wav][0], last_time)
        audio_dict[wav][1] = audio_dict[wav][1][:last_index]

    fs = 22050
    filter_bank = create_filter_bank(fs)
    # Calculate mfcc's
    for wav in audio_dict:
        print("MFCC's : " + wav)
        _fs, data = audio_dict[wav]
        mfccs = compute_mfccs(filter_bank, data, window_size)
        # print(mfccs.shape)
        # print(mfccs)
        # ymin = np.min(mfccs)
        # ymax = np.max(mfccs)
        # print(ymin)
        # print(ymax)
        # return
        # f = plt.figure()
        fig, ax = plt.subplots(1,1)

        num_windows = mfccs.shape[1]

        img = ax.imshow(mfccs, norm=LogNorm(vmin=1.0, vmax=1e9))
        ax.set_xticks([0, int(0.25 * num_windows), int(0.5*num_windows), int(0.75*num_windows), num_windows])
        x_label_list = [0, int(0.25*last_time), int(0.5*last_time), int(0.75*last_time), last_time]

        ax.set_xticklabels(x_label_list)
        # plt.imshow(mfccs)
        
        plt.xlabel("time (seconds)")
        plt.ylabel("filter #")
        title = wav[:-4] + " MFCC's"
        plt.title(title)
        # fig.colorbar(img, orientation="horizontal")
        # im = ax.matshow(C, cmap=cm.gray_r, norm=LogNorm(vmin=0.01, vmax=1))
        plt.gca().invert_yaxis()
        plt.show()
    
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

def test_no_clipping(): # TODO move these to another source file
    """
    Demonstrates no clipping when using a tapered window
    """
    print("Test avoiding clipping")
    last_time = 24
    window_size = 2048
    fft_size = 1000 #?

    # Load wavs
    fs, audio_sig = wavfile.read("wavs/chroma.wav")
    computed_audio = np.array(())

    num_bins = int(2 * (audio_sig.size / window_size))
    # for i in range(num_bins-1):
    for i in range(2):
        N = int(window_size/2)
        start_index = i* int(window_size/2)
        end_index = start_index + window_size
        signal_window = np.float64(audio_sig[start_index:end_index].copy())
        
        signal_window *= np.hamming(window_size)
        # print(signal_window)
        # plt.plot(np.blackman(window_size))
        # plt.plot(abs(signal_window))
        # plt.show()  

        # dtft = fft_window(signal_window).reshape((window_size,1))
        ffts = abs(np.fft.fft(signal_window))
        fft_mean = (np.mean(ffts)) * np.ones(ffts.shape)
        plt.plot(fft_mean[:1024])
        plt.plot(ffts[:1024])
        plt.show()

        audio = np.fft.ifft(ffts)
        audio = audio.reshape((audio.size, 1))

        if computed_audio.size == 0:
            computed_audio = audio
        else:
            # print(computed_audio.shape)
            # print(audio.shape)
            computed_audio[-N:] = computed_audio[-N:] + audio[:N]
            # print(computed_audio.shape)
            # print(audio.shape)
            computed_audio = np.append(computed_audio, audio[N:])
            computed_audio = computed_audio.reshape((computed_audio.size,1))
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.plot(computed_audio)
            # ax2 = fig.add_subplot(211)
            # ax2.plot(audio_sig[:end_index])
            # plt.show()
            # print("---")
    print(audio_sig.size)
    print(computed_audio.size)
    scaled = np.int16(computed_audio.real/np.max(np.abs(computed_audio.real)) * 32767)
    write("normal.wav", fs, scaled)

def test_find_peak_semitones():
    fs, audio_sig = wavfile.read("wavs/track201-classical.wav")
    fs = 22050
    compute_max_frequencies(audio_sig, fs)

def test_create_filter():
    create_filter_bank(22050, 40)

def problem2():
    print("Problem 2")
    last_time = 24
    window_size = 2048

    audio_dict = {}

    # Load wavs
    for wav in listdir("wavs"):
        if wav == "chroma.wav": # ignore chroma
            pass
        elif "wav" in wav: # make sure it's a wav file
            fs, data = wavfile.read("wavs/" + wav)
            audio_dict[wav] = [fs*2, data] # wavfile.read returns fs/2 (for some reason...)

    # Start out with just chroma
    # fs, data = wavfile.read("wavs/chroma.wav")
    # audio_dict["chroma.wav"] = [fs*2, data] # wavfile.read returns fs/2 (for some reason...)
    # fs, data = wavfile.read("wavs/track707-world.wav")
    # audio_dict["track707-world.wav"] = [fs*2, data] # wavfile.read returns fs/2 (for some reason...)
    # fs, data = wavfile.read("wavs/track463-metal.wav")
    # audio_dict["track463-metal.wav"] = [fs*2, data] # wavfile.read returns fs/2 (for some reason...)

    # Trim signals to 24s
    # for wav in audio_dict:
        # last_index = get_index_of_time(audio_dict[wav][0], last_time)
        # audio_dict[wav][1] = audio_dict[wav][1][:last_index]

    fs = 22050
    # weights = np.flipud(generate_pitch_weights(fs))
    weights = generate_pitch_weights(fs)
    # for i in range(weights.shape[1]):
    #     print(i)
    #     print(np.round(weights[:,i], 2))
    # fig, ax = plt.subplots(1,1)
    # img = ax.imshow(weights, interpolation='nearest', aspect='auto', cmap="plasma") #, norm=LogNorm(vmin=0.0001, vmax=1))
    # ax.set_ylim(0,11)
    # fig.colorbar(img)
    # plt.title("$f_s$ = 22050Hz weights matrix")
    # plt.title("$f_s$ = 2048Hz weights matrix")
    # plt.xlabel("FFT Index")
    # plt.ylabel("Note #")
    # plt.gca().invert_yaxis()
    # plt.show()
    # return

    print("\n...Ignore above warnings...\n")
    # Calculate mfcc's
    for wav in audio_dict:
        print("Getting pitch for: " + wav)
        _fs, data = audio_dict[wav]
        peak_freqs = compute_max_frequencies(data, fs)
        # for i in range(peak_freqs.shape[1]):
            # print(np.max(peak_freqs[:,i]))
            # print(len(np.nonzero(peak_freqs[:,i])))
            # print('---')
        # return
        # print(weights.shape)
        # print(peak_freqs.shape)
        pcp = np.dot( weights, peak_freqs)
        # for i in range(peak_freqs.shape[1]):
            # print(pcp[:,i])
        pcp[ pcp < 0.01] = 0.01
        # print(pcp.T)
        # print(pcp.shape)
        print(np.max(pcp))
        # print(np.min(pcp))


        # Plotting
        last_time = int(len(data) / fs)
        fig, ax = plt.subplots(1,1)
        num_windows = pcp.shape[1]
        img = ax.imshow(pcp, interpolation='nearest', aspect='auto', norm=LogNorm(vmin=0.01, vmax=1e14))
        ax.set_xticks([0, int(0.25 * num_windows), int(0.5*num_windows), int(0.75*num_windows), num_windows])
        ax.set_yticks(list(range(12)))
        x_label_list = [0, int(0.25*last_time), int(0.5*last_time), int(0.75*last_time), last_time]
        y_label_list = ['A', 'A#', 'B', "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        ax.set_yticklabels(y_label_list)
        ax.set_xticklabels(x_label_list)
        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("time (seconds)", fontsize=22)
        plt.ylabel("Note", fontsize=22)
        title = wav[:-4] + " PCP's"
        plt.title(title, fontsize=22)
        fig.colorbar(img) #, orientation="horizontal")
        # im = ax.matshow(C, cmap=cm.gray_r, norm=LogNorm(vmin=0.01, vmax=1))
        plt.gca().invert_yaxis()
        plt.show()

def problem5():
    print("Problem 5")
    window_size = 2048

    audio_dict = {}

    # Load wavs
    for wav in listdir("wavs"):
        if wav == "chroma.wav": # ignore chroma
            pass
        elif "wav" in wav: # make sure it's a wav file
            fs, data = wavfile.read("wavs/" + wav)
            audio_dict[wav] = [fs*2, data] # wavfile.read returns fs/2 (for some reason...)

    fs = 22050
    weights = generate_pitch_weights(fs)
    print("\n...Ignore above warnings...\n")
    # Calculate mfcc's
    for wav in audio_dict:
        print("Getting pcp's of: " + wav)
        _fs, data = audio_dict[wav]
        peak_freqs = compute_max_frequencies(data, fs)
        pcp = np.dot( weights, peak_freqs)
        pcp[ pcp < 0.01] = 0.01
        

if __name__ == "__main__":
    #test_fft()
    # test_clipping()
    # test_no_clipping()
    # problem1()
    # test_create_filter()
    # test_find_peak_semitones()
    # problem2()
    problem5()