import thinkdsp as dsp
import matplotlib.pyplot as plt
import numpy as np
import wavio
import scipy.signal as sig
import copy
import scipy.spatial.distance as dist

sound_speed = 1500

#The Following section contains definition of Helper functions
#to be used to write the code

def make_randombursts(sfs = 2000, fs = 96000, intervals = 8):
    """
    Assumes:
        sfs frequency of the carrier wave
        fs sampling frequency
        intervals number of random bursts to be created
    Returns:
        Wave object consisting of beeps with random amplitude
    """

    dt = 1/fs
    final = np.array([])

    for i in range(intervals):
        amp = 1
        wait = np.random.uniform(0.5,2)
        stime = np.random.uniform(0.02,0.1)
        t = np.arange(0,stime,dt)

        zeros = np.zeros(int(np.floor(fs*wait)))
        signal = amp*np.sin(t*np.pi*(1/stime))*np.sin(2*np.pi*t*sfs)
        final = np.concatenate((final,zeros))
        final = np.concatenate((final,signal))
        final = np.concatenate((final,zeros))
    final = np.concatenate((final, np.zeros(fs*2)))
    return dsp.Wave(final, framerate = fs)

def make_noiseburst(ntime = 0.1, stime = 0.5,fs = 96000, intervals = 20):
    """
    Assumes:
        ntime is the time of the noise
        stime is the of silence between bursts
        fs is the sampling frequency
        intervals is the number of noise bursts
    Returns:
        Wave object of noise bursts taken from a uniform distribution U~[0,1]
        with an "intervals" number and with an "ntime" time of sound spaced by "stime"
        seconds
    """
    ft = 1/fs
    final = np.array([])

    for i in range(intervals):
        noise = np.random.uniform(size = int(np.floor(fs*ntime)))
        silence = np.zeros(int(np.floor(fs*stime)))
        final = np.concatenate((final,silence))
        final = np.concatenate((final, noise))


    final = np.concatenate((final, silence))

    return final


def make_gausspulse(sfs ,iterations,fs):
    """Creates a wave object consisting of "iterations" number of gaussian modualted
    pulses with a carrier frequency of fs and a sampling rate of fs.
    """
    dt = 1/fs
    t = np.arange(-1,1,dt)

    sigma = 0.25**2

    y = np.exp(-1 * t**2/sigma**2)*np.sin(2*np.pi*sfs*t)

    final = y

    for i in range(iterations):
        final = np.concatenate((final,y))

    return final



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


def sdoa(wave1, wave2):
    """
    Returns :
        the number of samples of difference as calculated with the algorithm. If x is the result then this means
        that wave 1 arrived before by x samples (negative means it arrived later)
    Assumes:
        Wave1 and Wave2 are wave objects with same number of samples
        pass_band is the pass_band used for the filter. If nothing is passed the signals are not filtered.
    """

    #creates the wave objects with the new signals
    h1 = dsp.Wave(np.abs(Hilbert(wave1.ys)), framerate = wave1.framerate)
    h2 = dsp.Wave(np.abs(Hilbert(wave2.ys)), framerate = wave2.framerate)

    percentage = h1.duration * 0.001
    start = percentage
    duration = h1.duration - percentage

    segment1 = h1.segment(start = start, duration = duration)
    segment2 = h2.segment(start = start, duration = duration)

    return (-1 )* correlate(h1,h2)

def tdoa(wave1, wave2):
    """
    A simple wrapper for sdoa that changes units to seconds. Used only as a matter of
    convinicience. If x is the result this means that wave1 arrived x seconds before wave2. That is,
    if t_o is arrival of wave 1 and and t_2 is arrival of wave 2. t = t_2 - t_o
    """
    samples = sdoa(wave1, wave2)
    solution = (samples/wave1.framerate)

    return solution



def hplot(wave1, wave2):
    """
    Assummes
        wave1 and wave2 are wave objects
    Returns
        Plot of wave1 and wave2 with respective envelopes obtained with hilbert Transforms
        wave1 is in blue wave2 is in red. Envelope 1 is in black envelope 2 is in cyan
    """

    #Remember to modify


    h1 = dsp.Wave(np.abs(Hilbert(wave1.ys)), framerate = wave1.framerate)
    h2 = dsp.Wave(np.abs(Hilbert(wave2.ys)), framerate = wave2.framerate)


    plt.plot(wave1.ts, wave1.ys, color = 'b')
    plt.plot(wave2.ts, wave2.ys, color = 'r')
    plt.plot(h1.ts, h1.ys, color = 'k')
    plt.plot(h2.ts, h2.ys, color = 'c')

    plt.show()

#The following section conatins the definition of the classes to be used.

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
        self.sourcepos = np.array([0, 0, 0])


    def get_micarray( self ):
        return  self.mic_array

    def get_passband( self ):
        return self.passband

    def get_sourcepos( self ):
        return self.sourcepos.tolist()
    def get_mic_number( self ):
        """Returns the number of microphones in the array"""
        return len(self.micarray)

    def set_file(self, filename):
        """Sets the name of the file to analyze and initializes the wave instance
                variables in the microphone objects.
            Assumes: filename is a string in wav format and uses the specified passband
        """
        for mic in self.micarray:
            mic.read_wave( filename )

    def set_sourcepos(self, pos):
        """Sets the position of the source manually"""
        self.sourcepos = pos

    def apply_filter(self, range):
        """Applies the the band_pass filter with the set pass_band to
            each of the microphones"""
        for mic in self.micarray:
            mic.apply_filter(range = range)

    def find_loc(self):
        '''
        the algorithm obtains the position of the source creating sound using
        the method taken from stack exchange
        @https://math.stackexchange.com/questions/1722021/trilateration-using-tdoa
        '''
        mat_a = []
        mat_b = []
        #assumed anchor hydrophone always the first one in the array
        reference = self.micarray[0]
        reference_pos = reference.get_position()

        for i in np.arange(1, len(self.micarray)):
            mic = self.micarray[i]
            pos = self.micarray[i].get_position()
            ddoa = sound_speed * tdoa(mic.get_wave(),reference.get_wave())
            mat_a.append([reference_pos[0]-pos[0], reference_pos[1]-pos[1], ddoa])
            mat_b.append((reference_pos[0]**2 - pos[0]**2 + reference_pos[1]**2 - pos[1]**2 + ddoa**2)/2)

        mat_a = np.array(mat_a)
        mat_b = np.array(mat_b)
        b = mat_b.transpose()

        A = mat_a
        At = A.transpose()
        inv = np.linalg.inv(np.matmul(At, A))
        solution = np.matmul( np.matmul(inv, At), b)

        pos = np.array([solution[0], solution[1]])

        self.sourcepos = pos

    def plot(self, error = 2, show = True, save = False, additional = [],  title = "plot.png"):
        """
        Plots the current position of the microphones and the estimated position of the source
        Assumes:
            error: Number of meters that are given as error in the measurement of the sourcepos
            show: true by default. Indicates whether the plot is to be shown at the end of the function
            save: by default false. Indicates whether the plot is to be saved in the current directory
            title: title that will be used to save the file if save = True
            additional: The coordinates for an additional point to be plotted
        """
        #A list of Artists to put in the legend
        art = []
        labels = []

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        #Defines the properties of the circle  representing the source both the error
        #and the actual predicted location
        source_error = plt.Circle( (self.sourcepos[0], self.sourcepos[1]), radius = error)
        source_error.set_color('g')
        source_error.set_alpha(0.3)
        source_error.set_edgecolor('k')
        ax.add_patch(source_error)
        art.append(source_error)
        labels.append("Source error")

        source = plt.Circle( (self.sourcepos[0], self.sourcepos[1]), radius = 0.01)
        source.set_color('g')
        ax.add_patch(source)
        art.append(source)
        labels.append("Source")


        if(len(additional) != 0):
            other= plt.Circle( (additional[0], additional[1]), radius = 0.02)
            other.set_color('b')
            ax.add_patch(other)
            art.append(other)
            labels.append("Real location")


        ax.add_patch(source)

        for mic in self.micarray:
            pos = mic.get_position()
            dot = plt.Circle(( pos[0], pos[1]), radius = 0.01)
            dot.set_color( 'k' )
            ax.add_patch( dot )

        if(len(self.micarray) != 0):
            art.append(dot)
            labels.append("Hydrophones")


        #Make the graph pretty
        ax.autoscale_view()
        ax.legend(tuple(art), tuple(labels), loc = 'best', fontsize = 'x-small')
        plt.axis('equal')
        plt.xlabel("meters")
        plt.ylabel("meters")


        if(save):
            plt.savefig(title, dpi = 350)
        if(show):
            plt.show()






class Microphone:
    def __init__(self, position=[0.0, 0.0, 0.0], channel=1):
    	self.position = np.array(position)
    	self.channel = channel
    	self.wave= None


    def set_wave(self, wave):
        """sets a wave directly"""
        self.wave =  wave


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

    def get_position( self ):
        return self.position

    def get_channel( self ):
        return self.channel

    def get_wave( self ):
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
        self.wave.ys = self.butter_bandpass_filter(data = y, lowcut = range[0], highcut = range[1],fs = self.wave.framerate, order = 5)


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

    def get_position(self):
        return self.location


    def read_file(self , filename='sound.wav'):
        """
        Reads a file with the original signal and sets the instance variable
        to the array
        Assumes:
            filename is a string and the file it denotes contains only one channel

        """
        self.signal = dsp.read_wave(filename = filename)
        self.apply_padding()

    def set_wave(self, wave ):
        """
        Sets the wave with the neccesary padding at the front and at the back
        that was established by the file
        """
        self.signal = wave
        self.apply_padding()

    def apply_padding( self ):
        """
        Adds the neccesary padding to the signal instance variable.
        The variable used for the padding is that of the object when it is
        instantiated.
        """
        temp = self.signal.ys

        #padding entry and exit time
        pad_front = np.zeros(int(self.signal.framerate * self.tentry))
        pad_back = np.zeros(int(self.signal.framerate * self.texit))
        temp = np.concatenate((np.concatenate((pad_front, temp)), pad_back))

        self.signal = dsp.Wave(temp, framerate= self.signal.framerate)


    def emit(self, sensor):
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
            mic = sensor.micarray[i]
            d = dist.euclidean(self.location, mic.get_position())
            dt = d/sound_speed
            ds = int(dt * self.signal.framerate)
            final_waves[i].roll(ds)
            final_waves[i].ys =  final_waves[i].ys + np.random.normal(scale = self.noiselevel, size = len(self.signal.ys))

            mic.set_wave(final_waves[i])
