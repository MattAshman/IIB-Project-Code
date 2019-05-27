from scipy import signal
import pandas as pd
from FeatureExtraction import feature_tools
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pdb

plt.style.use('default')

# Used to find S2 pulse using coupling_interval
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_segment(x, s, v, sr, coupling_interval=0, plot_segment=True):
    # Get S1/S2 pulses from s
    s1s2_idxs = feature_tools.get_s1s2(s, sr, False)

    # Get ventricular activity locations from v
    v_idxs = feature_tools.get_vactivity(v, sr, False)

    # Identify S2 pulse
    s1s2_intervals = np.diff(s1s2_idxs)
    s2_idx = find_nearest(s1s2_intervals, coupling_interval)
    s2_idx = s1s2_idxs[s2_idx+1]
    s1_idxs = np.asarray([i for i in s1s2_idxs if ((i>50) & (i<s2_idx))])
    # Add small offset
    s1_idxs += 15
    s2_idx += 15

    # Identify ventricular activity after S2 pulse
    try:
        v_idx = v_idxs[np.argwhere(v_idxs > s2_idx)][0][0]
    except IndexError:
        print("Could not find ventricular activity index.")
        v_idx = None

    # For now just use strict segment length of 150.
    s1_segs = []    # List of S1 segments
    s2_seg = x[s2_idx:(s2_idx+150)] # S2 segment
    for s1_idx in s1_idxs:
        s1_segs.append(x[s1_idx:(s1_idx+150)])


    if plot_segment:
        fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16,4))
        ax1.plot(range(len(x)), x, range(s2_idx, s2_idx+150), s2_seg, 'g')
        ax1.hold(True)
        for i, s1_seg in enumerate(s1_segs):
            ax1.plot(range(s1_idxs[i], (s1_idxs[i]+150)), s1_seg, 'r')
        ax2.plot(range(len(s)), s)
        for idx in s1s2_idxs:
            ax2.axvline(idx, ymin=min(s), ymax=max(s), c='k', linestyle='--')
        plt.draw()
        plt.waitforbuttonpress()
        plt.close(fig)

    return np.array(s1_segs).astype(float), s2_seg, v_idx
