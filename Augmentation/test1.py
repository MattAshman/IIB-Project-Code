import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.arima_model as tsa
import parser
import data_augmentation
import pdb
from FeatureExtraction import feature_tools, get_segment

fileLoc = '/Users/matthewashman/github/MasterProject2018/Data/dataset01/'
files = ['0270', '0280', '0290', '0300', '0310', '0320', '0330', '0340', '0350', '0360', '0370', '0380']
features = np.zeros([len(files),2])
for fileIdx in range(len(files)):
    # Extract data from file
    file = files[fileIdx]
    fileName = fileLoc + 'TESTEXPORT' + file + '.txt'
    data, sr, numSamples = parser.parseFile(fileName)

    # Use the following channels
    v = data['V1']
    s = data['H 7-8']
    x = data['H 1-2']

    # Get segment
    seg = get_segment.get_segment(x, s, v, sr, False)
    seg = seg.values
    seg = np.asarray(seg)

    # Test data augmentation
    data_augmentation.augment_transformation(seg, True)
    data_augmentation.augment_noise_vector(seg, True)
