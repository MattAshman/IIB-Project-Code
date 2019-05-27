import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import copy
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    v = data['V1'].values
    v = v[:-800]    # Trim for visualisation
    s = data['H 7-8'].values
    s = s[:-800]    # Trim for visualisation
    x = data['H 1-2'].values
    x = x[:-800]    # Trim for visualisation
    x_sf = max(abs(x))

    # Get S1/S2 indeces
    s1s2_idxs, s_ma = feature_tools.get_s1s2(s, True)

    # Position axis for plot
    delta = (len(s)-len(s_ma))
    if (delta%2) == 0:
        start_idx = delta/2
        end_idx = len(s)-start_idx
    else:
        start_idx = int(np.floor(delta/2))
        end_idx = len(s)-start_idx-1

    s_range = range(len(s))
    s_ma_range = range(start_idx, end_idx)

    # Get ventricular activity indeces
    v_idxs, vf = feature_tools.get_vactivity(v, sr, debug=True)

    # Plot segment of interest
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16,4))
    ax1.plot(s_range, s/max(abs(s)), s_ma_range, s_ma/max(abs(s_ma)), zorder=1)
    ax1.set_ylabel('CSp', rotation=0, fontsize=13)
    ax1.legend(['CSp', 'Convolved Signal'], loc="upper right")
    ax3.plot(range(len(v)), v/max(abs(v)), range(len(vf)), vf/max(abs(vf)), zorder=1)
    ax3.set_ylabel('V1', rotation=0, fontsize=13)
    ax3.legend(['V1', 'Transformed Signal'], loc="upper right")
    ax2.plot(range(len(x)), x/max(abs(x)), zorder=1)
    ax2.set_ylabel('CS 1-2', rotation=0, fontsize=13)

    if v_idxs[0] < s1s2_idxs[0]:
        v_idxs = v_idxs[1:] # Remove first item

    if s1s2_idxs[-1] > v_idxs[-1]:
        s1s2_idxs = s1s2_idxs[:-1] # Remove last item

    for i, idx in enumerate(s1s2_idxs):
        ax1.axvline(idx, ymin=-1, ymax=1, c='k', linestyle='--', zorder=0, clip_on=False)
        if i < (len(s1s2_idxs)-1):
            ax1.text(idx+20, max(s)/max(abs(s))+0.05, 'S1')
        else:
            ax1.text(idx+20, max(s)/max(abs(s))+0.05, 'S2')
        ax2.axvline(idx, ymin=0.1
        , ymax=1, c='k', linestyle='--', zorder=0, clip_on=False)
        soi = patches.Rectangle((idx,-1),v_idxs[i]-idx,2,linewidth=1,edgecolor='k',facecolor='r', alpha=0.3)
        ax2.add_patch(soi)

    for i in v_idxs:
        ax2.axvline(i, ymin=-1, ymax=1, c='k', linestyle='--', zorder=0, clip_on=False)
        ax3.axvline(i, ymin=-1, ymax=1, c='k', linestyle='--', zorder=0, clip_on=False)

    for ax in [ax1, ax2, ax3]:
        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Hide the spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()


    plt.figure(figsize=(16,4))
    plt.plot(range(len(x)), x/x_sf)
    plt.hold(True)
    plt.plot()
    for i, idx in enumerate(s1s2_idxs):
        plt.axvline(x=idx, color='k', linestyle='--')
        if i < (len(s1s2_idxs)-1):
            plt.text(idx, max(x/x_sf)+0.1, 'S1', horizontalalignment='left')
        else:
            plt.text(idx, max(x/x_sf)+0.1, 'S2', horizontalalignment='left')
        plt.hold(True)
    plt.ylim([min(x/x_sf)-0.15, max(x/x_sf)+0.15])
    plt.axis('off')
    plt.tight_layout()

    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
