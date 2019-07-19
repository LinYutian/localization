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


class Microphone:
	def __init__(self, location=[0.0, 0.0, 0.0], channel=1, wave=null):
		self.position = position
		self.channel = channel
		self.wave=dsp.read_wave(filename=filename, channel = self.channel)

	def setwave(filename='sound.wav'):
		wave = dsp.read_wave(filename=filename, channel = self.channel)

	def apply_filter(pass_band = []):
		'''
		assumed always using the band_pass filter
		'''
		self.wave.band_pass(range = pass_band)

class Emitter:
	def __init__(self, location=[0.0, 0.0, 0.0], filename ='sound.wav', noiselevel=0, tentry=0, texit=0):
		self.location = location
		self.filename = filename
		self.noiselevel = noiselevel
		self.tentry =tentry
		self.texit = texit

	def new_file(filename='sound.wav'):
		signal = dsp.read_wave(filename = filename)

		#padding entry and exit time
		pad_front = np.zeros(signal.framerate * tentry)
		pad_back = np.zeros(signal.framerate * texit)
		np.concatenate([pad_front, signal, pad_back])

		#add noise
		



