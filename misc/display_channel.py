import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parser
import sys

plt.style.use('ggplot')

patients = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
numRecordings = 5

fileLoc = './Data/dataset00/'
patient = input('Choose patient. For example, "a". \n')

while patient not in patients:
    patient = input('Choose patient. For example, "a". \n')

channel = input('Choose channel. For example, "CS3-4". \n')

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=numRecordings, sharex=True)
plt.suptitle('Patient: ' + patient + ' Channel: ' + channel)

for recording in range(1, numRecordings+1):
    fileName = fileLoc + 'TESTEXPORT' + patient + '0' + str(recording) + '.txt'
    print('Extracting data from ' + fileName)
    data, samplingRate, numSamples = parser.parseFile(fileName)
    try:
        channelData = data[channel]
    except:
        sys.exit()
    plt.subplot(numRecordings, 1, recording)
    plt.plot(channelData)
    plt.ylabel(str(recording))

plt.xlabel('Sample')
plt.show()
