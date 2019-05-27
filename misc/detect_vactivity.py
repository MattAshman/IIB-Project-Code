from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parser
import sys
import detect_peaks
import features

plt.style.use('ggplot')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# User extract electrogram data
# fileLoc = './Data/dataset00/'
# file = input('Choose input file. For example, "e03". \n')
# fileName = fileLoc + 'TESTEXPORT' + file + '.txt'
# data, samplingRate, numSamples = parser.parseFile(fileName)

# Extract all electrogram data
fileLoc = './Data/dataset00/'
for patient in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
    for fileNum in ['1', '2', '3', '4', '5']:
        fileName = fileLoc + 'TESTEXPORT' + patient + '0' + fileNum + '.txt'
        data, samplingRate, numSamples = parser.parseFile(fileName)

        # Only interested in V1 for now
        V1 = data['V1']
        V1sq = np.square(V1)
        V1f =  butter_bandpass_filter(V1, 0.5, 125, samplingRate)
        V1fsq = np.square(V1f)
        mph = 0.3*max(V1fsq)  # Minimum peak height
        pulseIdx = detect_peaks.detect_peaks(V1fsq, mph=mph, mpd=200)

        fig = plt.figure()
        plt.plot(V1)
        plt.title(fileName)
        plt.ylabel('V1')
        plt.xlabel('Sample')
        plt.draw()
        plt.waitforbuttonpress()
        plt.close(fig)

        fig = plt.figure()
        plt.plot(V1f)
        plt.title(fileName)
        plt.ylabel('V1 Filtered')
        plt.xlabel('Sample')
        plt.draw()
        plt.waitforbuttonpress()
        plt.close(fig)

        fig = plt.figure()
        plt.plot(V1fsq)
        plt.title(fileName)
        plt.ylabel('V1 Filtered and Squared')
        plt.xlabel('Sample')
        plt.draw()
        plt.waitforbuttonpress()
        plt.close(fig)

        fig = plt.figure()
        plt.plot(V1.index, V1, pulseIdx, V1[pulseIdx], 'kx')
        plt.title(fileName)
        plt.ylabel('V1')
        plt.xlabel('Sample')
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()

#         # Only interested in CS data for now
#         cs = data[data.columns[-5:]]
#
#         # Identify pulses using rolling averages
#         csDiff = cs.diff()
#         csPulses = pd.rolling_mean(abs(csDiff), 50)
#         csPulses = pd.rolling_mean(abs(csPulses), 50)
#         csPulses = pd.rolling_mean(abs(csPulses), 20)
#         pulseIdxs = []  # Store pulse indexes
#         fig = plt.subplots(nrows=5)
#         plt.suptitle(fileName)
#         for colIdx, col in enumerate(csPulses):
#             colData = csPulses[col] # Extract column data
#             colData = colData.dropna()
#             startIdx = colData.first_valid_index()
#
#             mph = 0.5*max(colData)  # Minimum peak height
#             pulseIdx = detect_peaks.detect_peaks(colData, mph=mph)
#             pulseIdx = pulseIdx + startIdx/2
#
#             plt.subplot(5, 1, (colIdx+1))
#             plt.plot(cs[col].index.values, cs[col]/max(abs(cs[col])), colData.index.values, colData/max(abs(colData)))
#             plt.hold(True)
#             for idx in pulseIdx:
#                 plt.axvline(x=idx, color='k', linestyle='--')
#             pulseIdxs.append(pulseIdx)
#             plt.ylabel(col)
#
#         for ax in plt.gcf().axes:
#             try:
#                 ax.label_outer()
#             except:
#                 pass
#         plt.xlabel('Sample')
#         plt.draw()
#         plt.waitforbuttonpress(0) # this will wait for indefinite time
#         plt.close()
#
#         pulseIntervals = np.diff(pulseIdxs[-1])
#         s2Idx = np.argmin(pulseIntervals)
#         s1 = np.mean(pulseIntervals[0:s2Idx])
#         s2 = pulseIntervals[s2Idx]
#         print(fileName + ' - S1: ' + str(s1) + ' S2: ' + str(s2))
#
#         fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(nrows=5, sharex=True)
#         plt.suptitle(fileName)
#         for colIdx, col in enumerate(cs):
#             plt.subplot(5, 1, (colIdx+1))
#             plt.plot(cs[col].index.values, cs[col])
#             plt.hold(True)
#             for idx in pulseIdxs[-1]:
#                 plt.axvline(x=idx, color='k', linestyle='--')
#             plt.ylabel(col)
#         for ax in plt.gcf().axes:
#             try:
#                 ax.label_outer()
#             except:
#                 pass
#         plt.xlabel('Sample')
#         plt.draw()
#         plt.waitforbuttonpress(0) # this will wait for indefinite time
#         plt.close()
#
# # cs.plot(subplots=True)
# # plt.draw()
# # plt.waitforbuttonpress(0) # this will wait for indefinite time
#
#
# # csPulses.plot(subplots=True)
# # plt.hold()
# # plt.waitforbuttonpress(0) # this will wait for indefinite time
# # plt.close()
