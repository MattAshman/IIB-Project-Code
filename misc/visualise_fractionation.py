import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parser
import sys

plt.style.use('ggplot')

fileLoc = './Data/dataset01/'

# 1: View all data files. 2: View specific data file. 3: Compare H 1-2 progression. 4: Compare H 3-4 progression.
option = input('Viewing option: "1", "2", "3", "4". \n')

if option == '1':
    for interval in ['380', '370', '360', '350', '340', '330', '320', '310', '300', '290', '280', '270']:
        fileName = fileName = fileLoc + 'TESTEXPORT0' + interval + '.txt';
        try:
            data, samplingRate, numSamples = parser.parseFile(fileName)
        except:
            print("Couldn't open file.")
            sys.exit()
        data.plot(subplots=True, sharex=True)
        plt.suptitle(fileName)
        plt.xlabel('Sample')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()

elif option == "2":
    file = input('Choose S1S2 interval. For example, "280". \n')
    fileName = fileLoc + 'TESTEXPORT0' + file + '.txt'

    try:
        data, samplingRate, numSamples = parser.parseFile(fileName)
    except:
        print("Couldn't open file.")
        sys.exit()

    data.plot(subplots=True, sharex=True)
    plt.suptitle(fileName)
    plt.xlabel('Sample')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1)
    plt.show()

elif option == '3':
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=8, sharex=True)
    plt.suptitle('Channel: H 1-2')
    i = 1
    for interval in  ['380', '360', '340', '320', '300', '290', '280', '270']:
        fileName = fileName = fileLoc + 'TESTEXPORT0' + interval + '.txt';
        try:
            data, samplingRate, numSamples = parser.parseFile(fileName)
        except:
            print("Couldn't open file.")
            sys.exit()
        try:
            channelData = data['H 1-2']
        except:
            sys.exit()
        plt.subplot(8, 1, i)
        plt.plot(channelData)
        plt.ylabel(interval)
        i = i+1
    plt.xlabel('Sample')
    plt.show()

elif option == '4':
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=8, sharex=True)
    plt.suptitle('Channel: H 3-4')
    i = 1
    for interval in  ['380', '360', '340', '320', '300', '290', '280', '270']:
        fileName = fileName = fileLoc + 'TESTEXPORT0' + interval + '.txt';
        try:
            data, samplingRate, numSamples = parser.parseFile(fileName)
        except:
            print("Couldn't open file.")
            sys.exit()
        try:
            channelData = data['H 3-4']
        except:
            sys.exit()
        plt.subplot(8, 1, i)
        plt.plot(channelData)
        plt.ylabel(interval)
        i = i+1
    plt.xlabel('Sample')
    plt.show()

else:
    print('Invalid option. \n')
    sys.exit()
