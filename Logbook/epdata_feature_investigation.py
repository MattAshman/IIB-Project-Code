import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import copy
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from epdata_tools import epdata_main, get_original_ep_features
from fastdtw import fastdtw

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
X_at = epdata_main(file_loc, at_patients, 'at', list(reversed(coupling_intervals)))
X_avnrt = epdata_main(file_loc, avnrt_patients, 'avnrt', list(reversed(coupling_intervals)))
X_anrt = epdata_main(file_loc, avrt_patients, 'avrt', list(reversed(coupling_intervals)))
X_ep = epdata_main(file_loc, ep_patients, 'ep', list(reversed(coupling_intervals)))

X = pd.concat([X_af, X_at, X_avnrt, X_anrt, X_ep], ignore_index=True)

with open('/Users/matthewashman/github/MasterProject2018/AF_labels.csv', 'r') as file_in:
    labels = pd.read_csv(file_in)

labels = labels.dropna(how='all')
X['Label'] = np.nan
for idx, row in labels.iterrows():
    patient = row['Patient']
    channel = row['Channel']
    coupling_interval = str(int(row['Coupling Interval']))
    patient_type = patient[:1].lower()
    patient_num = patient[2:]
    if ((row['Label'] == '0') or (row['Label'] == '1')):
        X.loc[(X['Type']=='af') & (X['Patient']==patient_num) & (X['Channel']==channel) & (X['Coupling Interval']==coupling_interval) & (X['S1/S2']=='S2'), 'Label'] = row['Label']
    else:
        X.loc[(X['Type']=='af') & (X['Patient']==patient_num) & (X['Channel']==channel) & (X['Coupling Interval']==coupling_interval) & (X['S1/S2']=='S2'), 'Label'] = '-1'

X.to_pickle('/Users/matthewashman/github/MasterProject2018/Data/X_all.pkl')

pdb.set_trace()

X = pd.read_pickle('/Users/matthewashman/github/MasterProject2018/Data/X_af.pkl')

channel_names = ('CS1-2', 'CS3-4', 'CS5-6', 'CS7-8')
patient_type = 'af'
af_patients = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10') # AF patients
for patient in af_patients:
    for channel_name in channel_names:
        s2_data = X[(X['Type']==patient_type) & (X['Patient']==patient) & (X['Channel']==channel_name) & (X['S1/S2']=='S2')].reset_index()
        s1_data = X[(X['Type']==patient_type) & (X['Patient']==patient) & (X['Channel']==channel_name) & (X['S1/S2']=='S1')].reset_index()

        # # Plot S1/S2 responses
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
        #
        # fig, axes = plt.subplots(nrows=s2_data.shape[0], ncols=1, sharex=True, figsize=(4,8))
        # fig.suptitle(patient_type + ' Patient ' + patient + ' channel ' + channel_name + ' S2 response.')
        # for idx, row in s2_data.iterrows():
        #     axes[idx].plot(row['Data'])
        #     axes[idx].set_ylabel(row['Coupling Interval'])
        #
        # plt.subplots_adjust(left=0.25, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.1)
        # plt.draw()
        # plt.waitforbuttonpress()
        # plt.close()

        # Store features in s2_features
        s2_features = np.zeros([s2_data.shape[0], 6])
        s2_labels = np.zeros(s2_data.shape[0])
        s1_seg = s1_data.iloc[0]['Data']
        normal_feature_vec = np.append(get_original_ep_features(s1_seg), 0)
        for idx,s2 in s2_data.iterrows():
            # Get DWT distance first
            s2_seg = s2['Data']
            distance, path = fastdtw(s2_seg, s1_seg)

            s2_feature_vec = get_original_ep_features(s2_seg)
            s2_feature_vec = np.append(s2_feature_vec, distance)
            s2_feature_vec -= normal_feature_vec
            s2_features[idx,:] = s2_feature_vec
            s2_labels[idx] = s2['Label']

        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(8,6))
        [ax1, ax2, ax3, ax4, ax5, ax6] = axes.flatten()
        ax1.scatter(np.zeros(s2_data.shape[0]), s2_features[:,0], c=s2_labels, alpha=0.5)
        ax1.set_title('Number of Peaks')
        ax2.scatter(np.zeros(s2_data.shape[0]), s2_features[:,1], c=s2_labels, alpha=0.5)
        ax2.set_title('Conduction Delay')
        ax3.scatter(np.zeros(s2_data.shape[0]), s2_features[:,2], c=s2_labels, alpha=0.5)
        ax3.set_title('Activation Duration')
        ax4.scatter(np.zeros(s2_data.shape[0]), s2_features[:,3], c=s2_labels, alpha=0.5)
        ax4.set_title('Sample Entropy')
        ax5.scatter(np.zeros(s2_data.shape[0]), s2_features[:,4], c=s2_labels, alpha=0.5)
        ax5.set_title('Percentage Fractionation')
        ax6.scatter(np.zeros(s2_data.shape[0]), s2_features[:,5], c=s2_labels, alpha=0.5)
        ax6.set_title('DTW Distance')
        fig.suptitle(patient_type + 'patient' + patient + ' channel ' + channel_name)
        fig.subplots_adjust(hspace=0.3, wspace=0.4)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
