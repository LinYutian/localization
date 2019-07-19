import myThinkdsp as dsp
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


def Hilbert(signal):
    padding = np.zeros(int(2 ** np.ceil(np.log2(len(signal)))) - len(signal))
    tohilbert = np.hstack((signal, padding))

    result = sig.hilbert(tohilbert)

    result = result[0:len(signal)]

    return result

def correlate(wave1, wave2):
    """Assumes:
            wave1 and wave2 are wave objects from thinkdsp
            wave1 is the reference wave and wave2 is the wave to be shifted
    Returns:
            number of samples by which wave2 should be shifted to obtain max correlation ValueError
    Notes:
        correlation is calculated using FFT
    """
    corr = sig.correlate(wave1.ys, wave2.ys, mode = 'same', method='fft')
    max = np.argmax(corr) - len(corr)//2
    return max


def sdoa(wave1, wave2,pass_band = []):
    """
    Returns :
        the number of samples of difference as calculated with the algorithm. If x is the result then this means
        that one needs to shift wave 2 by x samples (positive means to the right, negative to the left) so that one obtains the maximum
        correlation. Hilbert transform is used
    Assumes:
        Wave1 and Wave2 are wave objects with same number of samples
        pass_band is the pass_band used for the filter. If nothing is passed the signals are not filtered.
    """
    if (len(pass_band) != 0):
        wave1.band_pass(pass_band)
        wave2.band_pass(pass_band)


    #creates the wave objects with the new signals
    h1 = dsp.Wave(np.abs(Hilbert(wave1.ys)), framerate = wave1.framerate)
    h2 = dsp.Wave(np.abs(Hilbert(wave2.ys)), framerate = wave2.framerate)

    percentage = h1.duration * 0.001
    start = percentage
    duration = h1.duration - percentage

    segment1 = h1.segment(start = start, duration = duration)
    segment2 = h2.segment(start = start, duration = duration)

    return correlate(h1,h2)

def hplot(wave1, wave2, pass_band = []):
    """
    Assummes
        wave1 and wave2 are wave objects
        pass_band is the pass_band to pass on to the filter. If empty no filter is applied
    Returns
        Plot of wave1 and wave2 with respective envelopes obtained with hilbert Transforms
        wave1 is in blue wave2 is in red. Envelope 1 is in black envelope 2 is in cyan
    """

    if(len(pass_band) != 0):
        wave1.band_pass(range = pass_band)
        wave2.band_pass(range = pass_band)

    h1 = dsp.Wave(np.abs(Hilbert(wave1.ys)), framerate = wave1.framerate)
    h2 = dsp.Wave(np.abs(Hilbert(wave2.ys)), framerate = wave2.framerate)


    plt.plot(wave1.ts, wave1.ys, color = 'b')
    plt.plot(wave2.ts, wave2.ys, color = 'r')
    plt.plot(h1.ts, h1.ys, color = 'k')
    plt.plot(h2.ts, h2.ys, color = 'c')

    plt.show()


class SensorArray:
    """Object to represent and Array of Sensors
    Provides the functuonality to localize
    """
    def __init__(self, mic_array, pass_band, soundfile):
        """Assumes: mic_array is a list of microphone objects
                    pos_source indicates the initial position of source and is
                        initialized to (0,0,0) by default
                    pass_band is a list of two elements indicating lower
                        and upper bounds"""

        self.micarray = mic_array
        self.passband = pass_band
        self.sourcepos = np.array([0, 0, 0])
        self.file = ""


    def get_micarray():
        return  self.mic_array

    def get_passband():
        return get_pass_band

    def get_sourcepos():
        return self.sourcepos.tolist()

    def get_file():
        return self.file

    def set_file(filename):
        """Sets the name of the file to analyze and initializes the wave instance
                variables in the microphone objects and applies the filtering.
            Assumes: filename is a string in wav format and uses the specified passband
        """
        self.file = filename
        for mic in micarray:
            mic.

    def filter_all():
        """Applies a band pass filter to all of the microphones in the list"""



    def plot():
