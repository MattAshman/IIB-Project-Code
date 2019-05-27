import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import features
import parser

plt.style.use('ggplot')
#plt.rcParams['axes.color_cycle'] = ['7fc97f', 'beaed4', 'fdc086']

def findS2(x):
    x = x/max(abs(x))   # Normalise the data
    startIdx = np.argmax(abs(x)>0.1) - 5
    endIdx = startIdx + 200

    return startIdx, endIdx

fileName = './Data/dataset00/TESTEXPORTa01.txt'

# data, samplingRate, numSamples = parser.parseFile(fileName)
# csData = data[data.columns[-5:]]
numRecordings = 5
patient = 'i'
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=numRecordings, sharex=True)
for recording in range(1, numRecordings+1):
    fileName = './Data/dataset00/TESTEXPORT' + patient + '0' + str(recording) + '.txt'
    data, samplingRate, numSamples = parser.parseFile(fileName)
    plt.subplot(numRecordings, 1, recording)
    plt.plot(data['CS3-4'])

plt.show()
# fig = plt.figure()
# data.plot(subplots = True)
# plt.waitforbuttonpress(0) # this will wait for indefinite time


# for col in csData:
#     startIdx, endIdx = findS2(csData[col].iloc[1550:])
#
#     segmentOI = csData[col].iloc[startIdx:endIdx]   # Segment of interest
#
#     fig = plt.figure()
#     plt.plot(range(numSamples), csData[col], 'k', range(startIdx, endIdx), segmentOI, 'r')
#     plt.title(col)
#     plt.xlabel('Sample')
#     plt.ylabel('Amplitude')
#     plt.legend(['Input data', 'Detected S2 pulse'])
#     plt.draw()
#     plt.waitforbuttonpress(0) # this will wait for indefinite time
#     plt.close(fig)
#
#     peakIdx, peakAmp = features.getPeakFeatures(segmentOI, 0.1)
#
#     fig = plt.figure()
#     plt.plot(range(startIdx, endIdx), segmentOI, 'r', (peakIdx+startIdx), peakAmp, 'kx')
#     for i, Idx in enumerate(peakIdx):
#         y = peakAmp[i]
#         timeDuration = str(Idx/samplingRate) + 's';
#         Idx = Idx + startIdx
#         plt.annotate(s='', xy=(startIdx,y), xytext=(Idx,y), arrowprops=dict(arrowstyle='<-', color='k'))
#     plt.title(col)
#     plt.xlabel('Sample')
#     plt.ylabel('Amplitude')
#     plt.legend(['Detected S2 pulse', 'Detected peaks'])
#     plt.draw()
#     plt.waitforbuttonpress(0) # this will wait for indefinite time
#     plt.close(fig)
