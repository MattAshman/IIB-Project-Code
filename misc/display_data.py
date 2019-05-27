import pandas as pd
import matplotlib.pyplot as plt
import parser
import sys

plt.style.use('seaborn-paper')

def display_data(fileLoc):
    file = input('Choose input file. For example, "e03" or "0270". \n')
    fileName = fileLoc + 'TESTEXPORT' + file + '.txt'

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
