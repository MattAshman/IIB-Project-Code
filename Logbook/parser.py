import pandas as pd
import pdb

def parseFile(fileName):
    with open(fileName, 'r') as fileIn:
        columnNames = []
        numSamples = None
        samplingRate = None
        lineCount = 0
        for line in fileIn:
            lineCount = lineCount + 1   # Increment line counter
            if(('Samples per channel:' in line) & (numSamples == None)):
                numSamples = int(line.split(': ', 1)[1].split('\n')[0]) # Set number of samples per channel
            if (('Sample Rate:' in line) & (samplingRate == None)):
                samplingRate = int(line.split(': ', 1)[1].split('Hz')[0])    # Set sampling rate
            if ('Label:' in line):
                columnNames.append(line.split(' ', 1)[1].split('\n')[0])    # Add channel label
            if ('[Data]' in line):
                break   # Data starts

    rawData = pd.read_csv(fileName, header = None, names = columnNames, delimiter = ',', skiprows = lineCount, encoding = "ISO-8859-1")

    return rawData, samplingRate, numSamples
