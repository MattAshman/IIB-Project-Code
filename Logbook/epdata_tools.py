import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import numpy as np
import pandas as pd
import parser
from FeatureExtraction import get_segment, feature_tools
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import time
import pdb


def epdata_main(file_loc, patients, patient_type, coupling_intervals):
    X_list = [] # Initially store data in a list of dictionaries
    column_names = ['Type', 'Patient', 'Coupling Interval', 'S1/S2', 'Data']    # Column names of pandas dataframe
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

            # Use the following channels
            try:
                v = data['V1'].values
            except KeyError:
                v = data['I'].values
            try:
                hisd = data['HISd'].values
                hisp = data['HISp'].values
            except KeyError:
                hisd = data['HIS d'].values
                hisp = data['HIS p'].values
            try:
                cs34 = data['CS 3-4'].values
                cs56 = data['CS 5-6'].values
                cs78 = data['CS 7-8'].values
            except KeyError:
                try:
                    cs34 = data['CS3-4'].values
                    cs56 = data['CS5-6'].values
                    cs78 = data['CS7-8'].values
                except KeyError:
                    print('Parsing error: could not extract CS3-4, CS5-6 and CS7-8 values for coupling interval ' + coupling_interval + 'ms.')
                    continue
            try:
                s = data['CSp'].values
                cs12 = data['CSd'].values
            except KeyError:
                try:
                    s = data['CS9-10'].values
                    cs12 = data['CS1-2'].values
                except KeyError:
                    try:
                        s = data['CS 9-10'].values
                        cs12 = data['CS 1-2'].values
                    except KeyError:
                        print('Parsing error: could not extract channel values for coupling interval ' + coupling_interval + 'ms.')
                        continue

            channel_names = ('CS1-2', 'CS3-4', 'CS5-6', 'CS7-8', 'CS9-10' 'HISp', 'HISd')
            channels = (cs12, cs34, cs56, cs78, cs910, hisp, hisd)
            if  all(isinstance(i, np.int64) for i in s) & all(isinstance(i, np.int64) for i in v):
                for channel, channel_name in zip(channels, channel_names):
                    if all(isinstance(i, np.int64) for i in channel):
                        # Get segment
                        s1_segs, s2_seg, v_idx = get_segment.get_segment(channel, s, v, sr, int(coupling_interval), False)
                        # Normalise segments
                        # s2_seg = s2_seg/max(abs(s2_seg))
                        temp_dict = {}
                        temp_dict.update({'Type': patient_type, 'Patient': patient, 'Coupling Interval': coupling_interval, 'Channel': channel_name, 'S1/S2': 'S2', 'Data': s2_seg})
                        X_list.append(temp_dict)
                        for i,s1_seg in enumerate(s1_segs):
                            temp_dict = {}
                            # s1_segs[i] = s1_seg/max(abs(s1_seg))
                            temp_dict.update({'Type': patient_type, 'Patient': patient, 'Coupling Interval': coupling_interval, 'Channel': channel_name, 'S1/S2': 'S1', 'Data': s1_segs[i]})
                            X_list.append(temp_dict)
                    else:
                        print('Parsing error: non-integer values in channel ' + channel_name + ' for coupling interval ' + coupling_interval + 'ms.')
            else:
                print('Parsing error: non-integer values in channels V1/CS9-10 for coupling interval ' + coupling_interval + 'ms.')
                continue

    X = pd.DataFrame(X_list)
    return X

def get_ep_features(x, sr=1000):
    feature_vec = feature_tools.get_features(x,sr)
    # dtw = fastdtw(x,z)
    # feature_vec = np.append(feature_vec, dtw[0])
    return feature_vec

def get_ep_feature_dict(x, col_name='', sr=1000):
    feature_dict = feature_tools.get_feature_dict(x, col_name)
    return feature_dict

def get_original_ep_features(x, sr=1000):
    feature_vec = np.zeros(5)
    feature_vec[0:4] = feature_tools.get_basic_features(x, sr)
    feature_vec[4] = feature_tools.sample_entropy(x, 2, 0.05)
    return feature_vec
