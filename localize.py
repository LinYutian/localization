import thinkdsp as dsp
import matplotlib.pyplot as plt
import numpy as np
import wavio
import scipy.signal as sig
import copy
import scipy.spatial.distace as dist

sound_speed = 1500



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
        that wave 1 arrived before by x samples (negative means it arrived later)
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

    return (-1 )* correlate(h1,h2)

def tdoa(wave1, wave2,speed = 1500, pass_band):
    """
    A simple wrapper for sdoa that changes units to seconds. Used only as a matter of
    convinicience. If x is the result this means that wave1 arrived x seconds before wave2
    """
    samples = sdoa(wave1, wave2, pass_band)
    return (samples/wave1.framerate)*speed



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
    def __init__(self, mic_array):
        """
        Assumes: mic_array is a list of microphone objects
                pos_source indicates the initial position of source and isinitialized to (0,0,0)
                by default
                soundfile is string indicating a file on the same folder. If an argument is passed
                then the constructor immidiately reads the soundfile in each of the arrays.
        """
        self.micarray = mic_array
        self.sourcepos = [0, 0, 0]


    def get_micarray():
        return  self.mic_array

    def get_passband():
        return get_pass_band

    def get_sourcepos():
        return self.sourcepos.tolist()
    def get_mic_number():
        """Returns the number of microphones in the array"""
        return len(self.micarray)


    def set_file(filename):
        """Sets the name of the file to analyze and initializes the wave instance
                variables in the microphone objects.
            Assumes: filename is a string in wav format and uses the specified passband
        """
        for mic in self.micarray:
            mic.set_wave( filename )

    def apply_filter():
        """Applies the the band_pass filter with the set pass_band to
            each of the microphones"""
        for mic in micarray:
            mic.apply_filter(range = range)

    def plot(error = 2, show = True, save = False, additional = [],  title = "plot.png"):
        """
        Plots the current position of the microphones and the estimated position of the source
        Assumes:
            error: Number of meters that are given as error in the measurement of the sourcepos
            show: true by default. Indicates whether the plot is to be shown at the end of the function
            save: by default false. Indicates whether the plot is to be saved in the current directory
            title: title that will be used to save the file if save = True
            additional: The coordinates for an additional point to be plotted
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        #Defines the properties of the circle  representing the source
        source = plt.Circle( (sourcepos[0], sourcepos[1]), radius = 1)
        source.set_color('g')
        source.set_alpha(0.5)
        source.set_edgecolor('k')

        if(len(additional) != 0):
            other= plt.Circle( (additional[0], additional[1]), radius = 0.01)
            source.set_color('b')
            source.set_alpha(0.5)
            source.set_edgecolor('k')

        ax.add_patch(source)

        for mic in micarray:
            pos = mic.get_position()
            dot = plt.Cirlce(( pos[0], pos[1])), radius = 0.01)
            dot.set_color('k')


        ax.autoscale_view()
        ax.figure.canvas.draw()

        if(save):
            plt.savefig(title, dpi = 300)
        if(show):
            plt.show()





class Microphone:
	def __init__(self, position=[0.0, 0.0, 0.0], channel=1):
		self.position = np.array(position)
		self.channel = channel
		self.wave= None


	def read_wave(filename='sound.wav'):
        """
        Reads a file and creates a wave objec that is then assigned to the instance variable abs
        representing the wave
        """

        wavob = wavio.read(filename)
        nchannel = len(wavob.data[0])
        sampw = wavob.sampwidth
        data = wavob.data
        framerate = wavob.rate

        ys = data[:,0]

        if(nchannel >= 2):
            ys = data[:,self.channel -1]

        #ts = np.arange(len(ys)) / framerate
        wav = dsp.Wave(ys, framerate=framerate)
        wav.normalize()
        self.wave = wav

    def set_wave(wave):
        """
        Sets the wave instance variable to the wave object created
        Assumes:
            wave is a Wave object from the thinkdsp module
        """
    def get_position():
        return self.position

    def get_channel():
        return self.channel

    def get_wave():
        return self.wave

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos= sig.butter(order, [low, high], btype='band', output ='sos')
        return sos

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sig.sosfiltfilt(sos, data, axis=-1, padtype='odd', padlen=None)
        return y


    def apply_filter(self, range):
		'''
		assumed always using the band_pass filter
		'''
        y = self.wave.ys
        self.wave.ys = self.butter_bandpass_filter(data = y, lowcut = range[0], highcut = range[1],fs = self.framerate, order = 5)





class Emitter:
	def __init__(self, location= [0.0, 0.0, 0.0], noiselevel=0, tentry=0, texit=0):
        """
        Object that simulates an emmitter of sound through a channel.

        """

        #Wave object from class thinkdsp that represents the original signal before
        #it is transmitted to the channels
        self.signal =None

		self.location = np.array(location)
		self.noiselevel = noiselevel
		self.tentry =tentry
		self.texit = texit


	def read_file(filename='sound.wav'):
        """
        Reads a file with the original signal and sets the instance variable
        to the array
        Assumes:
            filename is a string and the file it denotes contains only one channel

        """
		self.signal = dsp.read_wave(filename = filename)
        self.apply_padding()

    def set_wave( wave ):
        """
        Sets the wave with the neccesary padding at the front and at the back
        that was established by the file
        """
        self.signal = wave
        self.apply_padding()

    def apply_padding():
        """
        Adds the neccesary padding to the signal instance variable.
        The variable used for the padding is that of the object when it is
        instantiated.
        """
        temp = self.signal.ys

		#padding entry and exit time
		pad_front = np.zeros(signal.framerate * tentry)
		pad_back = np.zeros(signal.framerate * texit)
		temp = np.concatenate((np.concatenate((pad_front, temp)), pad_back))

        self.signal = dsp.Wave(temp, framerate= self.signal.framerate)


    def emit(sensor):
        """
        The function directly alters the state of the sensor array by passing
        wave objects to the microphones corresponding to what they would observe
        if this were the real world.
        Assumes:
            sensor is a SensorArray object with the given parameters
        """

        final_waves = []
        for i in np.arange(sensor.get_mic_number()):
            final_waves.append(copy.deepcopy(self.signal))

        for i in np.arange(sensor.get_mic_number()):
            mic = sensor.mic_array[i]
            d = dist.euclidian(self.location, mic.get_position())
            dt = d/sound_speed
            ds = np.floor(dt * self.signal.framerate)
            final_waves[i].roll(ds)
            final_waves[i].ys =  final_waves[i].ys + np.random.normal(scale = noise_level, size = len(signal.ys))
