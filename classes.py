import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

FILENAME = 'multipleTones.wav'
fs,vector = wavfile.read(FILENAME)
#print(fs)
data = []
AUD_LENGTH = 3
winSize = 6000
winNum = int((fs * AUD_LENGTH)//winSize)



class Window():

    t = 1/fs
    sec = t*winSize


    def __init__(self, data, start, fs, winSize):
        self.fft = np.fft.fft(data)
        self.freq = np.linspace(0, 1 / self.t, winSize)
        self.start = start
        self.end = start + len(data)
        self.fftmax = np.abs(self.fft)[:winSize//2].max()
        self.prominentFreq = self.findProminentFreq()
        self.prominentPitch = self.findProminentPitch()

    def __str__(self):
        '''returns the time where the window starts and the pitch in formatted form'''
        return str((self.start/winSize)*self.sec)+'\n' +'Pitch is: ' + str(self.prominentPitch)


    def plotfft(self):
        '''
        Plots the fast forier transform

        The x axis is Frequency and the y axis is amplitude. Only the first half
        of the fft is plotted because thats where all the relevant info is.
        '''
        
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency [Hz]")
        plt.plot(self.freq[:winSize // 2], np.abs(self.fft)[:winSize // 2] * 1/winSize) # 1 / N is a normalization factor
        plt.show()

    


    def findProminentFreq(self):
        '''
        Appends all frequencies in the fft above a certain threshold to a list

        Looks at all the peaks in the fft and if the peak is larger than half the
        largest peak, it is considered a relevant freqency and appended to the list.
        '''
        templist = []
        for ind in range(winSize//2):
            if np.abs(self.fft)[ind] >= self.fftmax/2 and np.abs(self.fft)[ind-1] < np.abs(self.fft)[ind] and np.abs(self.fft)[ind+1] < np.abs(self.fft)[ind]:
                templist.append(self.freq[ind])
        return templist


    def findProminentPitch(self):
        '''
        Converts all the prominent frequencies to pitches in a new list.

        Looks through the list of prominent frequencies and appendes the corresponding
        pitch to a new list which this funtion returns
        '''
        templist = []
        for freq in self.prominentFreq:
            templist.append(self.freqToPitch(freq))
        return templist

    def freqToPitch(self, freq):
        '''
        Returns the appropriate pitch for a certain frequncy


        This funtion has the range of allowed frequencies for a certain pitch.
        It takes a frequency and returns the corresponding pitch.Only works with
        frequencies from 100.95 to 719.21 for now.

        Parameter freq: freq is the frequency that you want to convert to a pitch
        Preconditon: Freq is an float or integer
        '''
        if freq >=100.905 and freq <= 106.91:#103.82
            return ('G#2/Ab2')
        elif freq >106.91 and freq <= 113.27:#110
            return ('A2')
        elif freq >113.27 and freq <= 120.005:#116.54
            return ('A#2/Bb2')
        elif freq >120.005 and freq <= 127.14:#123.47
            return ('B2')
        elif freq >127.14 and freq <= 134.7:#130.81
            return ('C3')
        elif freq >134.7 and freq <= 142.71:#138.59
            return ('C#3/Db3')
        elif freq >142.71 and freq <= 151.19:#146.83
            return ('D3')
        elif freq >151.19 and freq <= 160.18:#155.56
            return ('D#3/Eb3')
        elif freq >160.18 and freq <= 169.71:#164.81
            return ('E3')
        elif freq >169.71 and freq <= 179.8:#174.61
            return ('F3')
        elif freq >179.8 and freq <= 190.49:#184.99
            return ('F#3/Gb3')
        elif freq >190.49 and freq <= 201.82:#195.99
            return ('G3')
        elif freq >201.82 and freq <= 213.82:#207.65
            return ('G#3/Ab3')
        elif freq >213.82 and freq <= 226.54:#220
            return ('A3')
        elif freq >226.54 and freq <= 240.01:#233.08
            return ('A#3/Bflat3')
        elif freq >240.01 and freq <= 254.28:#246.94
            return ('B3')
        elif freq >254.28 and freq <= 269.4:#261.62
            return ('C4')
        elif freq >269.4 and freq <= 285.42:#277.18
            return ('C#4/Db4')
        elif freq >285.42 and freq <= 302.39:#293.66
            return ('D4')
        elif freq >302.39 and freq <= 320.37:#311.12
            return ('D#4/Eb4')
        elif freq >320.37 and freq <= 339.42:#329.62
            return ('E4')
        elif freq >339.42 and freq <= 359.60:#349.22
            return ('F4')
        elif freq >359.60 and freq <= 380.99:#369.99
            return ('F#4/Gb4')
        elif freq >380.99 and freq <= 403.64:#391.99
            return ('G4')
        elif freq >403.64 and freq <= 427.65:#415.30
            return ('G#4/Ab4')
        elif freq >427.65 and freq <= 453.08:#440
            return ('A4')
        elif freq >453.08 and freq <= 480.02:#466.16
            return ('A#4/Bb4')
        elif freq >480.02 and freq <= 508.56:#493.88
            return ('B4')
        elif freq >508.56 and freq <= 538.80:#523.25
            return ('C5')
        elif freq >538.80 and freq <= 570.84:#554.36
            return ('C#5/Db5')
        elif freq >570.84 and freq <= 604.78:#587.32
            return ('D5')
        elif freq >604.78 and freq <= 640.75:#622.25
            return ('D#5/Eb5')
        elif freq >640.75 and freq <= 678.85:#659.25
            return ('E5')
        elif freq >678.85 and freq <= 719.21:#698.45
            return ('F5')
        else:
            return('idk')


def normalizeData():
    '''
    Normalizes the range of the data to be from 0-1

    Divides every value by the largest number and appends it to a list.
    '''
    maxnum = 0
    for x in vector:
        if abs(x) > maxnum:
            maxnum = abs(x)
    for i in range(len(vector)):
        data.append(vector[i]/maxnum)

def prepPlotDataW1(window,freqList,timeList, activePitches):
    '''
    Initializes the prominent frequencies and times of the first window by placing them in their appropriate lists for plotting.

    Establishes the nested list by putting the first window of data in Frquecy domain with the prominent frequencies in a nested list
    and their corresponding time values in a nested list also. Ex: ProminentFreq = [250,279,440] so freqList = [[250],[279],[440]] and
    timeList = [[0],[0],[0]]. This makes graphing very easy.

    Parameter window: Window object
    Precondition: widnow is an instance of the Window class

    Parameter freqList: List for all the prominent frequencies (Will be a nested list)
    Precondition: freqList is an empty list

    Parameter timeList: List for all the prominent times that correspond to the prominent frequencies (Will be a nested list)
    Precondtion: timeList is an empty list

    Parameter activePitches: Not a nested list. Just a list of all pitches that have been detected.
    Precondition: activePitches is an empty list

    '''
    for i in window.prominentFreq:
        freqList.append([i])
        timeList.append([0])
        activePitches.append(window.freqToPitch(i))

def prepPlotData(window,freqList,timeList,activePitches):
    '''
    Organizes the frquency and time data to allow for plotting.

    Plotting requires that the y values and x values are contained in two separate lists. This function places
    the prominent frequencies and times into their appropriate lists.

    Parameter window: Window object
    Precondition: widnow is an instance of the Window class

    Parameter freqList: Nested list for all the prominent frequencies
    Precondition: freqList is a list

    Parameter timeList: Nested list for all the prominent times that correspond to the prominent frequencies
    Precondtion: timeList is a list

    Parameter activePitches: Not a nested list. Just a list of all pitches that have been detected.
    Precondition: activePitches is a list
    '''
    mainList = freqList.copy()
    for currentFreq in window.prominentFreq:
        for i in range(len(mainList)):
            if window.freqToPitch(currentFreq) ==  window.freqToPitch(mainList[i][0]):
                freqList[i].append(currentFreq)
                timeList[i].append((window.start/winSize)*window.sec)
        checkNewPitch(window,freqList,timeList,currentFreq, activePitches)


def checkNewPitch(window,freqList,timeList,currentFreq, activePitches):
    '''
    Checks to see if there are any new pitches being detected. If there are, they get appended to
    freqList and active Pitches.

    Helps with formatting the data for plotting and makes sure any new pitches are added to the lists
    for plotting.

    Parameter window: Window object
    Precondition: widnow is an instance of the Window class

    Parameter freqList: Nested list for all the prominent frequencies
    Precondition: freqList is a list

    Parameter timeList: Nested list for all the prominent times that correspond to the prominent frequencies
    Precondtion: timeList is a list

    Parameter currentFreq: The current frequency to be checking if it is a new frequency
    Precondtion: currentFreq is a float or int

    Parameter activePitches: Not a nested list. Just a list of all pitches that have been detected.
    Precondition: activePitches is a list
    '''
    for currentPitch in window.prominentPitch:
            if currentPitch not in activePitches:
                freqList.append([currentFreq])
                timeList.append([(window.start/winSize)*window.sec])
                activePitches.append(window.freqToPitch(currentFreq))


def plotFreqDomain(timeList,freqList):
    '''
    Plots the frequency as the y axis and time as the x axis

    Plots the values in the form of a scatter plot and separates the different pitches by different colors

    Parameter freqList: Nested list for all the prominent frequencies
    Precondition: freqList is a list

    Parameter timeList: Nested list for all the prominent times that correspond to the prominent frequencies
    Precondtion: timeList is a list
    '''
    for ind in range(len(timeList)):
        plt.scatter(timeList[ind],freqList[ind])
    #plt.yticks([220,247,262,294,330,349,587],['A:220','B:247','C:262','D:294','E:330','F:349','D:587'])
    plt.yticks(np.arange(0, 400, step=10))
    plt.grid()
    plt.show()


def run(fs):
    '''
    Runs all the above funtions in their correct order and establishes the necessary vairables

    Parameter fs: the sample rate that the wave file is sampled at
    Precondition: fs is a float
    '''
    timeList = []
    freqList = []
    activePitches = []
    normalizeData()
    window = Window(data[0:winSize], 0, fs, winSize)
    print(window)
    prepPlotDataW1(window,freqList,timeList, activePitches)
    outfile = open('audioFile.txt','w+')
    
    for ind in range(winNum-1):
        window = Window(data[(ind+1)*winSize:(ind+2)*winSize], (ind+1)*winSize, fs, winSize)
        prepPlotData(window,freqList,timeList, activePitches)
        print(window)
        #window.plotfft()  
        outfile.write('Frequency: ' + str(window.prominentFreq)+'\n')
        outfile.write('Seconds: ' + str(window.sec*window.start/winSize)+'\n')
    outfile.close()
    plotFreqDomain(timeList,freqList)

run(fs)




