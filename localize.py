import myThinkdsp as dsp
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig



#The Following section contains definition of Helper functions
#to be used to write the code

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

def tdoa(wave1, wave2,speed = 1500, pass_band = [1900,2100]):
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

#The following section conatins the definition of the classes to be used.

class SensorArray:
    """Object to represent and Array of Sensors
    Provides the functuonality to localize
    """
    def __init__(self, mic_array, pass_band, soundfile = ""):
        """Assumes: mic_array is a list of microphone objects
                    pos_source indicates the initial position of source and is initialized to (0,0,0)
                    by default
                    pass_band is a list of two elements indicating lower and
                    upper bounds"""

        self.micarray = mic_array
        self.passband = pass_band
        self.sourcepos = [0, 0, 0]
        self.file = soundfile

        #Immidiately applies the filter to the microphoness
        if(self.file != ""):
            self.set_file(soundfile)

    def get_micarray( self ):
        return  self.mic_array

    def get_passband( self ):
        return self.passband

    def get_sourcepos( self ):
        return self.sourcepos.tolist()

    def get_file( self ):
        "Returns the name of the file with which "
        return self.file

    def set_file(self, filename):
        """Sets the name of the file to analyze and initializes the wave instance
                variables in the microphone objects and applies the filtering.
            Assumes: filename is a string in wav format and uses the specified passband
        """
        self.file = filename
        for mic in self.micarray:
            mic.set_wave(self.file)
        self.apply_filter()

    def set_sourcepos(self, pos):
        """Sets the position of the source manually"""
        self.sourcepos = pos

    def apply_filter(self):
        """Applies the the band_pass filter with the set pass_band to
            each of the microphones"""
        for mic in micarray:
            mic.apply_filter(range = range)

    def getemitter(self):
        set_file(filename=self.file)
        '''
        the algorithm to trileterate is taken from stack exchange
        @https://math.stackexchange.com/questions/1722021/trilateration-using-tdoa
        '''
        mat_a = []
        mat_b = []
        #assumed anchor hydrophone always the first one in the array
        anchor = get_position(self.micarray[0])
        for mic in self.micarray:
            pos=mic.get_position
            mat_a.append([anchor[0]-pos[0],
                        anchor[1]-pos[1],
                        scipy.spatial.distance.euclidean(anchor, pos)])
            mat_b.append([(anchor[0]**2-pos[0]**2)/2,
                        (anchor[1]**2-pos[1]**2)/2,
                        (scipy.spatial.distance.euclidean(anchor, pos)**2)/2])
        mat_a.pop(0)
        mat_b.pop(0)
        np.array(mat_a)
        np.array(mat_b)
        solution = np.matmul(matmul(inv(np.matmul(mat_a.transpose(),mat_a)),mat_a.transpose()),mat_b)

        sourcepos = solution[:len(solution)-1] 






    def plot(self, error = 2, show = True, save = False, additional = [],  title = "plot.png"):
        """
        Plots the current position of the microphones and the estimated position of the source
        Assumes:
            error: Number of meters that are given as error in the measurement of the sourcepos
            show: true by default. Indicates whether the plot is to be shown at the end of the func
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
        source_error = plt.Circle( (self.sourcepos[0], self.sourcepos[1]), radius = 1)
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
            other= plt.Circle( (additional[0], additional[1]), radius = 0.01)
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
    def __init__(self, position=[0.0, 0.0, 0.0], channel=1, filename = ""):
        self.position = position
        self.channel = channel
        #Remember that the next line of code needs to be changed in the future
        if(len(filename) == 0):
            self.wave = None
        else:
            self.wave=dsp.read_wave(filename=filename, channel = self.channel)

    #Remember that this needs to be changed in the future
    def read_wave(self, filename='sound.wav'):
    	self.wave = dsp.read_wave(filename=filename, channel = self.channel)

    def set_wave(self, wave):
        """sets a wave directly"""
        self.wave =  wave


    #Remember that this needs to be changed in the future
    def apply_filter(self, pass_band = []):
    	'''
    	assumed always using the band_pass filter
    	'''
    	self.wave.band_pass(range = pass_band)

    def get_position( self ):
        return self.position

    def get_channel( self ):
        return self.channel

    def get_wave( self ):
        return self.wave


class Emitter:
    def __init__(self, location=[0.0, 0.0, 0.0], filename ='sound.wav', noiselevel=0, tentry=0, texit=0):
    	self.location = location
    	self.filename = filename
    	self.noiselevel = noiselevel
    	self.tentry =tentry
    	self.texit = texit

    def new_file(self, filename='sound.wav'):
    	signal = dsp.read_wave(filename = filename)

    	#padding entry and exit time
    	pad_front = np.zeros(signal.framerate * tentry)
    	pad_back = np.zeros(signal.framerate * texit)
    	np.concatenate([pad_front, signal, pad_back])

    	#add noise
        #signal.ys = signal.ys + np.random.normal(scale = noise_level, size = len(signal.ys))
