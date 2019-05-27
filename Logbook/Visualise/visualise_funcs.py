import pandas as pd
import matplotlib.pyplot as plt
import parser
import sys
import os

plt.style.use('ggplot')

def display_data(fileLoc):
    file = input('Choose input file. For example, "e03" or "0270". \n')
    fileName = 'Data/' + fileLoc + '/TESTEXPORT' + file + '.txt'
    fileName = '/Users/matthewashman/github/MasterProject2018/' + fileName
    # fileName = os.path.join( os.getcwd(), '..', fileName )

    try:
        data, samplingRate, numSamples = parser.parseFile(fileName)
    except:
        print("Couldn't open file: ", fileName)
        sys.exit()

    data.plot(subplots=True, sharex=True)
    plt.suptitle(fileLoc + ": " + file)
    plt.xlabel('Sample')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1)
    plt.show()

def display_channel(fileLoc):
    if fileLoc == 'dataset00':

        patients = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']    # Patients in dataset00
        numRecordings = 5   # 5 recordings per patient

        # User chooses patient and channel of interest
        patient = input('Choose patient. For example, "a". \n')
        while patient not in patients:
            patient = input('Choose patient. For example, "a". \n')
        channel = input('Choose channel. For example, "CS3-4". \n')

        fig = plt.subplots(nrows=numRecordings, sharex=True)
        plt.suptitle('Patient: ' + patient + ' Channel: ' + channel)
        for recording in range(1, numRecordings+1):
            fileName = './Data/' + fileLoc + '/TESTEXPORT' + patient + '0' + str(recording) + '.txt'
            fileName = os.path.join( os.getcwd(), '..', fileName )
            print('Extracting data from ' + fileName)
            data, samplingRate, numSamples = parser.parseFile(fileName)
            try:
                channelData = data[channel]
            except:
                print('Error: invalid channel.')
                sys.exit()
            plt.subplot(numRecordings, 1, recording)
            plt.plot(channelData)
            plt.ylabel(str(recording))
        plt.xlabel('Sample')
        plt.show()

    elif fileLoc == 'dataset01':
        S1S2intervals = ['380', '370', '360', '350', '340', '330', '320', '310', '300', '290', '280', '270']  # Available intervals

        # User chooses channel and files of interest
        channel = input('Choose channel. For example, "H 1-2". \n')
        startS1S2 = input('Choose start S1S2 interval. i.e. "320". \n')
        while startS1S2 not in S1S2intervals:
            startS1S2 = input('Choose start S1S2 interval. i.e. "320". \n')
        endS1S2 = input('Choose end S1S2 interval. i.e. "320". \n')
        while endS1S2 not in S1S2intervals:
            endS1S2 = input('Choose end S1S2 interval. i.e. "320". \n')

        startS1S2idx = S1S2intervals.index(startS1S2)
        endS1S2idx = S1S2intervals.index(endS1S2)
        numPlots = endS1S2idx - startS1S2idx + 1

        fig = plt.subplots(nrows=numPlots, sharex=True)
        plt.suptitle('Channel: ' + channel)
        i = 1
        for interval in  S1S2intervals[startS1S2idx:(endS1S2idx+1)]:
            fileName = 'Data/' + fileLoc + '/TESTEXPORT0' + interval + '.txt'
            fileName = os.path.join( os.getcwd(), '..', fileName)
            print(fileName)
            try:
                data, samplingRate, numSamples = parser.parseFile(fileName)
            except:
                print("Couldn't open file.")
                sys.exit()
            try:
                channelData = data[channel]
            except:
                print('Invalid channel name. \n')
                sys.exit()
            plt.subplot(numPlots, 1, i)
            plt.plot(channelData)
            plt.ylabel(interval)
            i = i+1
        plt.xlabel('Sample')
        plt.show()
    else:
        print('Invalid file location.')
        sys.exit()

def compare_patients(fileLoc):
    if fileLoc == 'dataset00':
        patients = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']    # Available patients
        patient1, patient2 = [None, None]
        channelNames = ['CSd', 'CS3-4', 'CS5-6', 'CS7-8', 'CSp'] # Available channels
        channel = None
        numRecordings = 5   # Number of recordings per patient

        while patient1 not in patients:
            patient1 = input('Choose patient 1. For example, "a". \n')

        while patient2 not in patients:
            patient2= input('Choose patient 2. For example, "b". \n')

        while channel not in channelNames:
            channel = input('Choose channel. For example, "CS3-4". \n')

        channelIdx = channelNames.index(channel)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(numRecordings, 2, sharex=True)
        plt.suptitle('Patients: ' + patient1 + ' and ' + patient2 + '    Channel: ' + channel)

        # Loop through each patient
        for patientIdx in range(1, 3):
            if patientIdx == 1:
                patient = patient1
            else:
                patient = patient2

            for recording in range(1, numRecordings+1):
                fileName = 'Data/' + fileLoc + '/TESTEXPORT' + patient + '0' + str(recording) + '.txt'
                fileName = os.path.join( os.getcwd(), '..', fileName )
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
    else:
        print('Invalid dataset. \n')
        sys.exit()

def show_fractionatation():
    S1S2intervals = ['380', '370', '360', '350', '340', '330', '320', '310', '300', '290', '280', '270']  # Available intervals
    # User chooses channel and files of interest
    channel = input('Choose channel. For example, "H 1-2". \n')
    firstS1S2 = input('Choose first S1S2 interval. i.e. "320". \n')
    while firstS1S2 not in S1S2intervals:
        firstS1S2 = input('Choose first S1S2 interval. i.e. "320". \n')
    secondS1S2 = input('Choose second S1S2 interval. i.e. "320". \n')
    while secondS1S2 not in S1S2intervals:
        secondS1S2 = input('Choose second S1S2 interval. i.e. "320". \n')

    firstName = 'Data/' + 'dataset01' + '/TESTEXPORT0' + firstS1S2 + '.txt'
    secondName = 'Data/' + 'dataset01' + '/TESTEXPORT0' + secondS1S2 + '.txt'
    firstName = os.path.join( os.getcwd(), '..', firstName)
    secondName = os.path.join(os.getcwd(), '..', secondName)
    try:
        data1, samplingRate1, numSamples1 = parser.parseFile(firstName)
        data2, samplingRate2, numSamples2 = parser.parseFile(secondName)
    except:
        print("Couldn't open file.")
        sys.exit()
    try:
        channelData1 = data1[channel]
        channelData2 = data2[channel]
    except:
        print('Invalid channel name. \n')
        sys.exit()

    plt.plot(range(1345,1500), channelData1[1345:1500], range(1345,1500), channelData2[1345:1500], '--')
    plt.xlabel('Sample')
    plt.legend(['Normal Activity', 'Fractionated Activity'])
    plt.show()
