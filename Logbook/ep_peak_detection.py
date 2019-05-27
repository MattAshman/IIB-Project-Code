import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from epdata_tools import epdata_main
from FeatureExtraction import feature_tools

plt.style.use('default')

# N_aug = 5   # Number of augmented examples per file.
file_loc = '/Users/matthewashman/github/MasterProject2018/Data/EPdata/'
af_patients = ('2', '3', '4', '5', '6', '8', '9', '10') # AF patients
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

# all_patients = (af_patients, at_patients, avnrt_patients, avrt_patients, ep_patients)
# all_patients = (('1'), ('1'), ('1'), ('1'), ('1'))
# patient_types = ('af', 'at', 'avnrt', 'avrt', 'ep')
all_patients = (af_patients)
patient_type = 'af'
# for patients, patient_type in zip(all_patients, patient_types):
for patients in all_patients:
    for patient in patients:
        patient_data = X[(X['Type']==patient_type) & (X['Patient']==patient)]
        cs12_s2_data = patient_data[(patient_data['Channel']=='CS1-2') & (patient_data['S1/S2']=='S2')].reset_index()
        cs12_s1_data = patient_data[(patient_data['Channel']=='CS1-2') & (patient_data['S1/S2']=='S1')].reset_index()
        cs34_s2_data = patient_data[(patient_data['Channel']=='CS3-4') & (patient_data['S1/S2']=='S2')].reset_index()
        cs34_s1_data = patient_data[(patient_data['Channel']=='CS3-4') & (patient_data['S1/S2']=='S1')].reset_index()

        # Plot S1/S2 responses CS1-2
        fig, axes = plt.subplots(nrows=cs12_s1_data.shape[0], ncols=1, sharex=True, figsize=(6,8))
        fig.suptitle(patient_type + ' Patient ' + patient + ' channel CS1-2 S1 response.')
        for i, row in cs12_s1_data.iterrows():
            # Get peaks
            seg = row['Data']
            peak_idxs, peak_amps = feature_tools.get_peaks(seg, 0.2, None, False)
            axes[i].plot(seg)
            axes[i].hold(True)
            for j, idx in enumerate(peak_idxs):
                # axes[i].text(idx, (peak_amps[j]), str(i+1), horizontalalignment='left', verticalalignment='bottom', fontsize = 14)
                axes[i].plot(idx, peak_amps[j], 'rx')
                axes[i].hold(True)

            axes[i].set_ylabel(row['Coupling Interval'])

        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.1)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()

        fig, axes = plt.subplots(nrows=cs12_s2_data.shape[0], ncols=1, sharex=True, figsize=(6,8))
        fig.suptitle(patient_type + ' Patient ' + patient + ' channel CS1-2 S2 response.')
        for i, row in cs12_s2_data.iterrows():
            seg = row['Data']
            peak_idxs, peak_amps = feature_tools.get_peaks(seg, 0.2, None, False)
            axes[i].plot(seg)
            axes[i].annotate(s='', xy=(0, peak_amps[0]), xytext=(peak_idxs[0], peak_amps[0]), arrowprops=dict(arrowstyle='<->'))
            axes[i].text(round(peak_idxs[0]/2), (peak_amps[0]+0.05), 'Conduction delay', horizontalalignment='center', fontsize = 6)
            axes[i].hold(True)
            for j, idx in enumerate(peak_idxs):
                # axes[i].text(idx, (peak_amps[j]), str(i+1), horizontalalignment='left', verticalalignment='bottom', fontsize = 14)
                axes[i].plot(idx, peak_amps[j], 'rx')
                axes[i].hold(True)
            axes[i].set_ylabel(row['Coupling Interval'])

        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.1)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()

        # Plot S1/S2 responses CS3-4
        fig, axes = plt.subplots(nrows=cs34_s1_data.shape[0], ncols=1, sharex=True, figsize=(6,8))
        fig.suptitle(patient_type + ' Patient ' + patient + ' channel CS3-4 S1 response.')
        for i, row in cs34_s1_data.iterrows():
            # Get peaks
            seg = row['Data']
            peak_idxs, peak_amps = feature_tools.get_peaks(seg, 0.2, None, False)
            axes[i].plot(seg)
            axes[i].hold(True)
            for j, idx in enumerate(peak_idxs):
                # axes[i].text(idx, (peak_amps[j]), str(i+1), horizontalalignment='left', verticalalignment='bottom', fontsize = 14)
                axes[i].plot(idx, peak_amps[j], 'rx')
                axes[i].hold(True)
            axes[i].set_ylabel(row['Coupling Interval'])

        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.1)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()

        fig, axes = plt.subplots(nrows=cs34_s2_data.shape[0], ncols=1, sharex=True, figsize=(6,8))
        fig.suptitle(patient_type + ' Patient ' + patient + ' channel CS3-4 S2 response.')
        for i, row in cs34_s2_data.iterrows():
            # Get peaks
            seg = row['Data']
            peak_idxs, peak_amps = feature_tools.get_peaks(seg, 0.2, None, False)
            axes[i].plot(seg)
            axes[i].annotate(s='', xy=(0, peak_amps[0]), xytext=(peak_idxs[0], peak_amps[0]), arrowprops=dict(arrowstyle='<->'))
            axes[i].text(round(peak_idxs[0]/2), (peak_amps[0]+0.05), 'Conduction delay', horizontalalignment='center', fontsize = 6)
            axes[i].hold(True)
            for j, idx in enumerate(peak_idxs):
                # axes[i].text(idx, (peak_amps[j]), str(i+1), horizontalalignment='left', verticalalignment='bottom', fontsize = 14)
                axes[i].plot(idx, peak_amps[j], 'rx')
                axes[i].hold(True)
            axes[i].set_ylabel(row['Coupling Interval'])

        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.1)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
