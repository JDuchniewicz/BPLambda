from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq
from scipy.stats import skew, kurtosis
import numpy as np

# SAMPLING FREQUENCY
FS = 125

# TEMPORAL FEATURES

def time_cycle(inp, sys_1, sys_2, dia_1):
    # sometimes there may be no diastole to the left (no proper one) -> in cases which break the algorithm
    # (signal is not pure in the window), return -1 and then clean the signal
    THRESHOLD = 0.5 # safely basing on the data we can assume such minimal threshold for dia/sys difference
    dia_2 = np.argmin(inp[sys_1:sys_2]) + sys_1
    #print(dia_2 + sys_1)
    cycle_len = dia_2 - dia_1
    return cycle_len, dia_2

def dicr_notch(inp_ii, sys_1):
    peaks_raw, _ = find_peaks(inp_ii[sys_1:], height=0) # find first above 0 in the ppg'' after systole
    peaks_raw += sys_1
    if len(peaks_raw) < 1:
        return -1
    return peaks_raw[0]

def time_start_sys(sys_1, dia_1):
    return sys_1 - dia_1

def time_sys_end(sys_1, dia_2):
    return dia_2 - sys_1

def time_sys_dicr(sys_1, dicr):
    return dicr - sys_1

def time_dicr_end(dia_2, dicr):
    return dia_2 - dicr

def ratio_sys_dia(inp, sys_1, dia_1):
    return inp[sys_1] / inp[dia_1]

def extract_features(inp, inp_ii):
    # calculate the peaks -> systolic (maxima)
    peaks_raw, _ = find_peaks(inp)
    if len(peaks_raw) < 1: # hopeless if could not find any peaks
        return -1, -1, -1, -1, -1, -1
    sys_1 = peaks_raw[0]
    if len(peaks_raw) < 2:
        sys_2 = -1
    else:
        sys_2 = peaks_raw[1]

    # to find the beginning of the cycle (diastole), we need to go left of the first systole
    dia_1 = np.argmin(inp[:sys_1])

    cycle_len, dia_2 = time_cycle(inp, sys_1, sys_2, dia_1)
    #print(f"{sys_1} : {sys_2} : {dia_1} : {dia_2} : {cycle_len}")
    t_start_sys = time_start_sys(sys_1, dia_1)
    t_sys_end = time_sys_end(sys_1, dia_2)

    dicr = dicr_notch(inp_ii, sys_1) # if could not extract dicrotic -> abort, not worth it
    if dicr == -1:
        return -1, -1, -1, -1, -1, -1
    t_sys_dicr = time_sys_dicr(sys_1, dicr)
    t_dicr_end = time_dicr_end(dia_2, dicr)
    ratio = ratio_sys_dia(inp, sys_1, dia_1)

    cycle_len /= FS
    t_start_sys /= FS
    t_sys_end /= FS
    t_sys_dicr /= FS
    t_dicr_end /= FS
    return cycle_len, t_start_sys, t_sys_end, t_sys_dicr, t_dicr_end, ratio


# SPECTRAL FEATURES

def freq_fftfreq(inp):
    freq = fft(inp)
    fftfreqs = fftfreq(FS, 1 / FS)[:FS//2]
    return freq, fftfreqs

def freq_magnitudes(freq, fftfreqs):
    yf = freq
    xf = fftfreqs
    #plt.plot(xf, np.abs(yf[:FS//2])) # do not normalize (not to loose float precision?)
    #plt.show()
    freq_mag = zip(xf[:FS//2], np.abs(yf[:FS//2]))
    sort = sorted(freq_mag, reverse=True, key=lambda pair: pair[1])

    # always ignore DC component
    freqs, mags = [x[0] for x in sort[1:4]], [x[1] for x in sort[1:4]]
    return freqs, mags

def norm_energy(freq):
    fabs = np.abs(freq[:FS//2])
    energy = 0.
    for f in fabs:
        energy += f * f
    energy /= FS
    return energy

def fft_entropy(freq):
    # normalize the input
    freqs = freq / FS

    entr = 0.
    for f in freqs:
        entr += f * np.log(f)
    return -entr

def extract_spectral_features(inp):
    freq, fftfreqs = freq_fftfreq(inp)
    freqs, mags = freq_magnitudes(freq, fftfreqs)
    energy = norm_energy(freq)
    entro = fft_entropy(freq).real
    bins, _ = np.histogram(np.abs(freq) / np.sum(freq), bins=np.arange(0, FS, FS//10))
    skewness = skew(freq).real
    kurt = kurtosis(freq).real

    return freqs[0], mags[0], freqs[1], mags[1], freqs[2], mags[2], energy, entro, bins, skewness, kurt

# GRADIENTS
def gradients(ppg):
    ppg_i = np.gradient(ppg)
    ppg_ii = np.gradient(ppg_i)
    return ppg_ii

# MAIN EXTRACTION FUNCTION
def prepare_data(ppg):
	ppg_ii = gradients(ppg)

	# for now just 1 period for inference, if more, need for additional changes as in the training notebook
	cycle_len, t_start_sys, t_sys_end, t_sys_dicr, t_dicr_end, ratio = extract_features(ppg, ppg_ii)
	freq_1, mag_1, freq_2, mag_2, freq_3, mag_3, energy, entro, bins, skewness, kurt = extract_spectral_features(ppg)
	return np.array((cycle_len, t_start_sys, t_sys_end, t_sys_dicr, t_dicr_end, ratio,
				freq_1, mag_1, freq_2, mag_2, freq_3, mag_3, energy, entro, skewness, kurt, *bins))
