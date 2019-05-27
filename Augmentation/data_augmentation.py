import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import filters
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from FeatureExtraction import get_segment, feature_tools
import pdb

plt.style.use('default')

def augment_noise_vector(x, N, debug=False):
    # Get peaks
    peak_idxs, peak_amps = feature_tools.get_peaks(x, 0.05, None, False)

    # Find appropriate starting point
    try:
        start_idx = peak_idxs[3]
    except:
        try:
            start_idx = peak_idxs[1]
        except:
            start_idx = peak_idxs[0]

    max_length = len(x)-start_idx
    x_aug = np.zeros([N, len(x)])

    for i in range(0,N):
        # Get length and scale factor
        n_length = int(max(0,np.random.normal(60, 10)))
        n_length = min(max_length, n_length)
        n_sf = max(0.1,np.random.normal(0.25, 0.05))
        # Generate noise vector
        n =  n_sf*np.random.randn(n_length)

        # Clip end values to blend into segment
        n[-6:] = 0
        n[:6] = 0
        # Convolve with Gaussian to make smooth
        n = filters.gaussian_filter1d(n, np.random.normal(2.5, 0.5))

        # Add noise vector to create augmentated data
        x_aug[i,:] = x[:]/max(abs(x))
        x_aug[i, start_idx:(start_idx+n_length)] = x_aug[i, start_idx:(start_idx+n_length)] + n

    if ((debug==True) & (N>3)):
        fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(4,6))
        ax1.plot(x)
        ax1.axis('off')
        for i, ax in enumerate([ax2, ax3, ax4, ax5]):
            ax.plot(x/max(abs(x)), '--', alpha=0.5)
            ax.hold(True)
            ax.plot(x_aug[i,:]/max(abs(x_aug[i,:])))
            ax.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close(fig)

    return x_aug

def augment_fractionation(x, N, debug=False):
    # Jittering
    def jitter(x, sigma=0.05):
        my_noise = np.random.normal(loc=0, scale=sigma, size=len(x))
        return x+my_noise

    # Scaling
    def scaling(x, sigma=0.1):
        scaling_factor = np.random.normal(loc=1.0, scale=sigma)
        return x*scaling_factor

    def generate_random_curves(x, sigma=0.2, knot=50):
        xx = np.arange(0,len(x), (len(x)-1)/(knot+1))
        yy = np.random.normal(loc=1.0, scale=sigma, size=knot+2)

        x_range = np.arange(len(x))
        cs = CubicSpline(xx, yy)

        return np.array(cs(x_range))

    # Magnitude warping
    def magwarp(x, sigma):
        cs = generate_random_curves(x, sigma, knot=25)
        # Blend cs curve towards 1 at both ends
        # tau = 10
        # delta = int(np.floor(len(x)/2))
        # for i in range(delta):
        #     cs[i] = (1-np.exp(-i/tau))*cs[i] + np.exp(-i/tau)*0
        #     cs[-(i+1)] = (1-np.exp(-i/tau))*cs[-(i+1)] + np.exp(-i/tau)*0

        return x*cs

    def distort_timesteps(x, sigma=0.2):
        tt = generate_random_curves(x, sigma, knot=7) # Regard these samples aroun 1 as time intervals
        tt_cum = np.cumsum(tt)        # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        t_scale = (len(x)-1)/tt_cum[-1]
        tt_cum = tt_cum*t_scale
        return tt_cum

    # Time warping
    def timewarp(x, sigma=0.2):
        tt_new = distort_timesteps(x, sigma)
        x_new = np.zeros(len(x))
        x_range = np.arange(len(x))
        x_new = np.interp(x_range, tt_new, x)
        return x_new

    # Rotation
    def rotation(x):
        axis = np.random.uniform(low=-1, high=1, size=1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        return np.matmul(x , axangle2mat(axis,angle))


    # x = x/max(abs(x))   # Normalise
    x_aug = np.zeros([N, len(x)])

    # Create N augmented examples
    for i in range(0,N):
        x_aug[i,:] = timewarp(x, 0.2)
        # x_aug[i,:] = timewarp(x_aug[i,:], 0.2)
        # x_aug[i,:] = timewarp(x_aug[i,:], 0.2)
        # x_aug[i,:] = magwarp(x_aug[i,:], 0.2)
        x_aug[i,:] = magwarp(x_aug[i,:], 0.1)
        x_aug[i,:] = timewarp(x_aug[i,:], 0.2)
        x_aug[i,:] = magwarp(x_aug[i,:], 0.2)
        # x_aug[i,:] = x_aug[i,:]/max(abs(x_aug[i,:]))
        x_aug[i,:] = timewarp(x_aug[i,:], 0.2)
        x_aug[i,:] = magwarp(x_aug[i,:], 0.2)
        x_aug[i,:] = timewarp(x_aug[i,:], 0.2)
        x_aug[i,:] = magwarp(x_aug[i,:], 0.2)


    if ((debug==True) & (N>3)):
        fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(4,6))
        ax1.plot(x)
        ax1.axis('off')
        for i, ax in enumerate([ax2, ax3, ax4, ax5]):
            ax.plot(x/max(abs(x)), '--', alpha=0.5)
            ax.hold(True)
            ax.plot(x_aug[i,:]/max(abs(x_aug[i,:])))
            ax.axis('off')

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close(fig)

    return x_aug
