import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import numpy as np
import pandas as pd
import parser
import matplotlib.pyplot as plt
from itertools import cycle, islice

plt.style.use('default')

file_loc = '/Users/matthewashman/github/MasterProject2018/Data/EPdata/'
af_patients = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10') # AF patients
at_patients = ('1', '2', '3') # AT patients
avnrt_patients = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22') # AVNRT patients
avrt_patients = ('1', '2', '3', '4', '5', '6', '7', '8') # AVRT patients
ep_patients = ('1', '2', '3', '4', '5', '6', '7', '8') # EP patients
coupling_intervals = ('220','230', '240', '250', '260', '270', '280', '290', '300',
                      '310', '320', '330', '340', '350', '360', '370', '380', '390', '400')

patients = af_patients
patient_type = 'af'
for patient in patients:
    for coupling_interval in coupling_intervals:
        # Extract data from file
        # file_name = file_loc + 'TESTEXPORT' + file + '.txt'
        try:
            file_name = file_loc + patient_type + patient + '/' + patient_type + 'patient' + patient + '-0' + coupling_interval + '.txt'
            data, sr, num_samples = parser.parseFile(file_name)
        except FileNotFoundError:
            try:
                file_name = file_loc + patient_type + patient + '/' + patient_type + 'patient0' + patient + '-0' + coupling_interval + '.txt'
                data, sr, num_samples = parser.parseFile(file_name)
            except:
                print('Parsing error: coupling interval ' + coupling_interval + 'ms not available for patient ' + patient + '.')
                continue

        plot_colours = list(islice(cycle(['k']), None, len(data)))
        data[['CS1-2', 'CS3-4', 'CS5-6', 'CS7-8', 'CS9-10']].plot(subplots=True, color = 'k', sharex=True, grid=True, figsize=(16,5))
        plt.xlabel('Sample')
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.925, top=0.8, wspace=0.1, hspace=0.1)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
