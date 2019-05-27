import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import features
import parser

plt.style.use('ggplot')

def findS2(x):
    x = x/max(abs(x))   # Normalise the data
    startIdx = np.argmax(abs(x)>0.1) - 5
    endIdx = startIdx + 200

    return startIdx, endIdx

fileLoc = './Data/dataset00/'
patients = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
afPatients = ['a', 'b', 'c', 'd']
numRecordings = 5
s2Idx = 1550
# Choose which channel to compare
channelNames = ['CSd', 'CS3-4', 'CS5-6', 'CS7-8', 'CSp']
channelString = ', '.join(channelNames)
channel = input('Choose channel to compare. Available options are: ' + channelString + '.\n')
while channel not in (channelNames):
    channel = input('Choose channel to compare. Available options are: ' + channelString + '.\n')

channelIdx = channelNames.index(channel)

# Store two categories of data
healthyData = []
afData = []

for patient in patients:
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=numRecordings, sharex=True)
    plt.suptitle('Patient: ' + patient + ' Channel: ' + channel)
    patientData = []
    for recording in range(1, numRecordings+1):
        fileName = fileLoc + 'TESTEXPORT' + patient + '0' + str(recording) + '.txt'
        print('Extracting data from ' + fileName)
        data, samplingRate, numSamples = parser.parseFile(fileName)
        csData = data[data.columns[-5:]]

        channelData = csData.iloc[:, channelIdx]

        startIdx, endIdx = findS2(channelData.iloc[s2Idx:])
        segmentOI = channelData.iloc[startIdx:endIdx]   # Segment of interest
        peakIdx, peakAmp = features.getPeakFeatures(segmentOI, 0.1, fileName)
        patientData.append(peakIdx)

        plt.subplot(numRecordings, 1, recording)
        plt.plot(range(startIdx, endIdx), segmentOI, 'r', (peakIdx+startIdx), peakAmp, 'kx')
        # for i, Idx in enumerate(peakIdx):
        #     y = peakAmp[i]
        #     timeDuration = str(Idx/samplingRate) + 's';
        #     Idx = Idx + startIdx
        #     plt.annotate(s='', xy=(startIdx,y), xytext=(Idx,y), arrowprops=dict(arrowstyle='<-', color='k'))
    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)

    if patient in afPatients:
        afData.append(patientData)
    else:
        healthyData.append(patientData)

afData = np.array(afData)
healthyData = np.array(healthyData)
print(afData)
print(healthyData)
