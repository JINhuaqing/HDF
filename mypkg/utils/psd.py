import scipy.signal as signal
import numpy as np

# the two fns are from pverma: https://github.com/parulv1/reorder_ChangData/blob/main/utils/psd.py
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = signal.lfilter_zi(b, a) * data[0]
    y, _ = signal.lfilter(b, a, data, zi=zi)
    return y