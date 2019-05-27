import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from epdata_tools import epdata_main

plt.style.use('default')

# N_aug = 5   # Number of augmented examples per file.
file_loc = '/Users/matthewashman/github/MasterProject2018/Data/EPdata/'
af_patients = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10') # AF patients
at_patients = ('1', '2', '3') # AT patients
avnrt_patients = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22') # AVNRT patients
avrt_patients = ('1', '2', '3', '4', '5', '6', '7', '8') # AVRT patients
ep_patients = ('1', '2', '3', '4', '5', '6', '7', '8') # EP patients
coupling_intervals = ('220','230', '240', '250', '260', '270', '280', '290', '300',
                      '310', '320', '330', '340', '350', '360', '370', '380', '390', '400')

X_af = epdata_main(file_loc, af_patients, 'af', tuple(reversed(coupling_intervals)))
# X_at = epdata_main(file_loc, at_patients, 'at', list(reversed(coupling_intervals)))
# X_avnrt = epdata_main(file_loc, avnrt_patients, 'avnrt', list(reversed(coupling_intervals)))
# X_anrt = epdata_main(file_loc, avrt_patients, 'avrt', list(reversed(coupling_intervals)))
# X_ep = epdata_main(file_loc, ep_patients, 'ep', list(reversed(coupling_intervals)))

# X = pd.concat([X_af, X_at, X_avnrt, X_anrt, X_ep], ignore_index=True)
X = X_af

pdb.set_trace()

channel_names = ('HISp', 'HISd')

# all_patients = (af_patients, at_patients, avnrt_patients, avrt_patients, ep_patients)
patients = af_patients
# patient_types = ('af', 'at', 'avnrt', 'avrt', 'ep')
patient_type = 'af'
# for patients, patient_type in zip(all_patients, patient_types):
for patient in patients:
    for channel_name in channel_names:
        s2_data = X[(X['Type']==patient_type) & (X['Patient']==patient) & (X['Channel']==channel_name) & (X['S1/S2']=='S2')].reset_index()
        s1_data = X[(X['Type']==patient_type) & (X['Patient']==patient) & (X['Channel']==channel_name) & (X['S1/S2']=='S1')].reset_index()

        # Plot S1/S2 responses
        # fig, axes = plt.subplots(nrows=s1_data.shape[0], ncols=1, sharex=True, figsize=(4,8))
        # fig.suptitle(patient_type + 'patient' + patient + ' channel ' + channel_name + ' S1 response. ')
        # for idx, row in s1_data.iterrows():
        #     axes[idx].plot(row['Data'])
        #     axes[idx].set_ylabel(row['Coupling Interval'])
        #
        # plt.subplots_adjust(left=0.25, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.1)
        # plt.draw()
        # plt.waitforbuttonpress()
        # plt.close()

        fig, axes = plt.subplots(nrows=s2_data.shape[0], ncols=1, sharex=True, figsize=(4,8))
        fig.suptitle(patient_type + ' Patient ' + patient + ' channel ' + channel_name + ' S2 response.')
        for idx, row in s2_data.iterrows():
            axes[idx].plot(row['Data'])
            axes[idx].set_ylabel(row['Coupling Interval'])

        plt.subplots_adjust(left=0.25, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.1)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
