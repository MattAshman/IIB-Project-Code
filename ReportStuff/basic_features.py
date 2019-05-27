import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import copy
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parser
from FeatureExtraction import get_segment, feature_tools

plt.style.use('default')


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
    seg_sf = max(abs(seg))

    # Get peaks
    peak_idxs, peak_amps = feature_tools.get_peaks(seg, 0.05, None, True)

    plt.figure(figsize=(16,4))
    plt.plot(range(len(seg)), seg/seg_sf)
    plt.hold(True)
    plt.annotate(s='', xy=(0, peak_amps[0]/seg_sf), xytext=(peak_idxs[0], peak_amps[0]/seg_sf), arrowprops=dict(arrowstyle='<->'))
    plt.text(round(peak_idxs[0]/2), (peak_amps[0]/seg_sf+0.05), 'Conduction delay', horizontalalignment='center', fontsize = 14)
    plt.annotate(s='', xy=(peak_idxs[0], max(seg/seg_sf)+0.1), xytext=(peak_idxs[-1], max(seg/seg_sf)+0.1), arrowprops=dict(arrowstyle='<->'))
    plt.text(round((peak_idxs[0]+peak_idxs[-1])/2), max(seg/seg_sf)+0.13, 'Activity duration', horizontalalignment='center', fontsize = 14)
    for i, idx in enumerate(peak_idxs):
        plt.text(idx, (peak_amps[i]/seg_sf), str(i+1), horizontalalignment='left', verticalalignment='bottom', fontsize = 14)
        plt.plot(idx, peak_amps[i]/seg_sf, 'rx')
        plt.hold(True)

    plt.text(len(seg)-50, 0.3, 'Number of deflections: ' + str(len(peak_idxs)), horizontalalignment='center', fontsize = 14)
    plt.ylim([min(seg/seg_sf)-0.15, max(seg/seg_sf)+0.15])
    plt.axis('off')
    plt.grid()
    plt.tight_layout()

    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
