import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parser
import sys

plt.style.use('ggplot')

patients = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
patient1, patient2 = [None, None]
channelNames = ['CSd', 'CS3-4', 'CS5-6', 'CS7-8', 'CSp']
channel = None
numRecordings = 5

fileLoc = './Data/dataset00/'

while patient1 not in patients:
    patient1 = input('Choose patient 1. For example, "a". \n')

while patient2 not in patients:
    patient2= input('Choose patient 2. For example, "b". \n')

while channel not in channelNames:
    channel = input('Choose channel. For example, "CS3-4". \n')

channelIdx = channelNames.index(channel)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(numRecordings, 2, sharex=True)
plt.suptitle('Patients: ' + patient1 + ' and ' + patient2 + '    Channel: ' + channel)

for patientIdx in range(1, 3):
    print(patientIdx)
    if patientIdx == 1:
        patient = patient1
    else:
        patient = patient2
    for recording in range(1, numRecordings+1):
        fileName = fileLoc + 'TESTEXPORT' + patient + '0' + str(recording) + '.txt'
        print('Extracting data from ' + fileName)
        data, samplingRate, numSamples = parser.parseFile(fileName)
        csData = data[data.columns[-5:]]
        try:
            channelData = csData.iloc[:, channelIdx]
        except:
            sys.exit()
        plt.subplot(numRecordings, 2, (2*recording-(patientIdx%2)))
        plt.plot(channelData)
        plt.ylabel(str(recording))
    plt.xlabel('Sample')
plt.show()
