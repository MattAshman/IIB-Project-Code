import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.robust import mad
from tsfresh.feature_extraction import feature_calculators
from scipy.interpolate import interp1d

import pdb
import time

def denoise(x):
    # Obtain Daubechies N=6 wavelet coefficients
    waveletCoefs = pywt.wavedec(x, 'db6', mode='per')

    # Throw away coefficients corresponding to noise
    sigma = mad(waveletCoefs[-1])
    uThresh = 2*sigma*np.sqrt(2*np.log(len(x)))
    denoised = waveletCoefs[:]
    denoised[1:] = (pywt._thresholding.hard(i, value=uThresh) for i in denoised[1:])

    # Reconstruct the original signal
    xDenoised = pywt.waverec(denoised, 'db6', mode='per')

    return xDenoised

def get_peaks(x, height_thresh, scale_amp=None, set_scale=False, plot = False):
    x = np.array(x)

    # Get height_thresh
    if set_scale:
        height_thresh = height_thresh*scale_amp
    else:
        height_thresh = height_thresh*max(abs(x))

    # Denoise x
    xdn = denoise(x)

    # Detect peaks using detect_peaks
    pos_peak_idx = detect_peaks(xdn, mph=height_thresh, threshold = 0)
    neg_peak_idx = detect_peaks((-xdn), mph=height_thresh, threshold = 0)
    peak_idx = np.concatenate([pos_peak_idx, neg_peak_idx])
    peak_idx = np.sort(peak_idx)
    # Edge indeces aren't detected
    peak_idx = peak_idx[(peak_idx != 0) & (peak_idx != (len(xdn)-1))]

    new_peak_idx = []
    peak_amp = []
    if (len(peak_idx) > 0):
        new_peak_idx.append(peak_idx[0])
        mp_thresh = 0.2*max(abs(x))
        for i in range(len(peak_idx)-1):
            idx = peak_idx[i]
            idx_next = peak_idx[i+1]
            mid_point = int((idx_next+idx)/2)
            if (max([abs(x[idx_next]-x[mid_point]), abs(x[idx]-x[mid_point])]) > mp_thresh):
                new_peak_idx.append(idx_next)

        peak_idx = np.array(new_peak_idx)
        peak_amp = x[peak_idx]

    if plot == True:
        fig, [ax1] = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,8))
        ax1.plot(x, 'b' , xdn, 'r--', peak_idx, peak_amp, 'kx')
        #plt.title(fileName)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Normalised amplitude')
        ax1.legend(['Original segment', 'Denoised segment', 'Detected peaks'])

        plt.draw()
        plt.waitforbuttonpress(0) # this will wait for indefinite time
        plt.close(fig)


    return peak_idx, peak_amp

def detect_peaks(x, mph=None, maxph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks > maximum peak height
    if ind.size and maxph is not None:
        ind = ind[x[ind] <= maxph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dxmean = np.mean(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dxmean < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        detect_peaks_plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def detect_peaks_plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

def get_s1s2(s, sr=1000, debug=True):

    # def butter_lowpass_filter(data, lowcut, fs, order=5):
    #     nyq = 0.5*fs
    #     low = lowcut/nyq
    #     b,a = signal.butter(order, low, btype='high')
    #     y = signal.lfilter(b, a, data)
    #     return y
    #
    # # Denoise s first (remove low frequency)
    # s = butter_lowpass_filter(s, 0.1, sr)

    s1s2_idxs = []
    amp_thresh = max(s)
    for idx, val in enumerate(s):
        if val > 0.9*amp_thresh:
            if (len(s1s2_idxs)==0):
                s1s2_idxs.append(idx)
            elif(idx - s1s2_idxs[-1] > 150):
                s1s2_idxs.append(idx)

    s1s2_idxs = np.asarray(s1s2_idxs)

    if debug:   # Plot for debugging
        s_range = range(len(s))
        s_ma_range = range(start_idx, end_idx)
        plt.figure(figsize=(16,4))
        plt.plot(s_range, s/max(abs(s)), s_ma_range, s_ma/max(abs(s_ma)))
        plt.hold(True)
        # for i in s1s2_idxs:
        #     plt.axvline(x=i, color='k', linestyle='--')

        plt.axis('off')
        plt.legend(['CSp', 'Convolved Signal'])
        plt.tight_layout()
        plt.waitforbuttonpress()
        plt.close()

    return s1s2_idxs#, s_ma

def get_vactivity(v, sr, debug=True):

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        try:
            y = signal.lfilter(b, a, data)
        except:
            pdb.set_trace()
            print('wtf')
        return y

    # Apply series of transformations to bring out ventricular peaks
    vf =  butter_bandpass_filter(v, 0.5, 125, sr)
    vf = np.square(vf)

    mph = 0.3*max(vf)  # Minimum peak height
    v_idxs = detect_peaks(vf, mph=mph, mpd=200)
    v_idxs = v_idxs - 50  # Move to start of activity

    if debug:   # Plot for debugging
        plt.figure(figsize=(16,4))
        plt.plot(v/max(abs(v)))
        plt.hold(True)
        plt.plot(vf/max(abs(vf)))
        # for idx in v_idxs:
        #     plt.axvline(x=idx, color='k', linestyle='--')
        plt.axis('off')
        plt.legend(['V1', 'Transformed Signal'])
        plt.tight_layout()
        plt.waitforbuttonpress()
        plt.close()

    return v_idxs#, vf

def get_basic_features(x, sr, show_peaks=False):
    peak_idxs, peak_amps = get_peaks(x, 0.05, None, show_peaks)

    features = []

    num_peaks = len(peak_idxs); features.append(num_peaks)
    if (len(peak_idxs)>0):
        conduction_delay = peak_idxs[0]/sr
        activity_duration = (peak_idxs[-1]-peak_idxs[0])/sr
        prcnt_fractionation = percentage_fractionation(x, peak_idxs, sr)
    else:
        conduction_delay = 0
        activity_duration = 0
        prcnt_fractionation = 0

    features.append(conduction_delay); features.append(activity_duration); features.append(prcnt_fractionation)

    return np.asarray(features)

def get_features(x, sr, show_peaks=False):
    # # Get peaks
    # peak_idxs, peak_amps = get_peaks(x, 0.05, None, show_peaks)
    features = []
    #
    # num_peaks = len(peak_idxs); features.append(num_peaks)
    # if (len(peak_idxs)>0):
    #     conduction_delay = peak_idxs[0]/sr
    #     activity_duration = (peak_idxs[-1]-peak_idxs[0])/sr
    #     prcnt_fractionation = percentage_fractionation(x, peak_idxs, sr)
    # else:
    #     conduction_delay = 0
    #     activity_duration = 0
    #     prcnt_fractionation = 0
    #
    # features.append(conduction_delay); features.append(activity_duration); features.append(prcnt_fractionation)

    # Abritrary selection of features from TSFresh
    erbc = feature_calculators.energy_ratio_by_chunks(x, [{'num_segments':10, 'segment_focus':3}, {'num_segments':10, 'segment_focus':2}])
    features.append(erbc[0][1])
    features.append(erbc[1][1])
    features.append(feature_calculators.approximate_entropy(x, 2, 0.7))
    features.append(feature_calculators.approximate_entropy(x, 2, 0.5))
    features.append(feature_calculators.approximate_entropy(x, 2, 0.9))
    features.append(feature_calculators.ratio_beyond_r_sigma(x, 1))
    imq = feature_calculators.index_mass_quantile(x, [{'q': 0.6}, {'q': 0.4}])
    features.append(imq[0][1])
    features.append(imq[1][1])
    features.append(feature_calculators.kurtosis(x))
    # features.append(feature_calculators.sample_entropy(x))
    features.append(feature_calculators.binned_entropy(x, 10))
    features.append(feature_calculators.quantile(x,0.9))
    ar_coeffs = feature_calculators.ar_coefficient(x, [{'coeff': 1, 'k': 10}, {'coeff': 2, 'k': 10}])
    features.append(ar_coeffs[0][1])
    features.append(ar_coeffs[1][1])
    features.append(feature_calculators.first_location_of_minimum(x))
    features.append(feature_calculators.first_location_of_maximum(x))


    # energy = np.dot(x,x); features.append(energy)
    # mean_abs_change = np.mean((abs(np.diff(x)))); features.append(mean_abs_change)
    # # An estimate for a time series complexity.
    # cid_ce = np.sqrt(np.dot(np.diff(x), np.diff(x))); features.append(cid_ce)
    # msdc = feature_calculators.mean_second_derivative_central(x); features.append(msdc)
    # mean_abs = np.mean(np.abs(x)); features.append(mean_abs)
    # std = np.std(x); features.append(std)
    # prcnt_reoccurring = feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints(x); features.append(prcnt_reoccurring)
    # max = np.max(x); features.append(max)
    # min = np.min(x); features.append(min)

    return np.asarray(features)

def get_feature_dict(x, col_prefix='', thresh_cd=None, set_thresh_cd=False, thresh_peaks=None, set_thresh_peaks=False, show_peaks=False):
    feature_dict = {}

    # First two features is conduction delay. First use the typical threshold, then use threshold based on segment.
    # Ambiguity regarding the definition of conduction delay.
    if set_thresh_cd:
        feature_dict[col_prefix + 'Conduction Delay: set_thresh=True'] = get_delay(x, thresh_cd, set_thresh_cd)
        feature_dict[col_prefix + 'Conduction Delay: set_thresh=False'] = get_delay(x)
    else:
        feature_dict[col_prefix + 'Conduction Delay'] = get_delay(x)

    # Features 3 and 4 are number of peaks. First use typical threshold, then use threshold based on segment.
    # The issue with using the number of peaks as a feature is that it is extremelly sensitive to errors. Also
    # lots of abiguity with regards to what defines a peak. Should a peak be defined relative to a universal
    # threshold, or a local threshold?
    height_thresh=0.1
    if set_thresh_peaks:
        peaks = get_peaks(x, height_thresh, thresh_peaks, set_thresh_peaks, plot=False)
        feature_dict[col_prefix + 'Number of Peaks: set_thresh=True'] = len(peaks[0])
        peaks = get_peaks(x, height_thresh)
        feature_dict[col_prefix + 'Number of Peaks: set_thresh=False'] = len(peaks[0])
    else:
        peaks = get_peaks(x, height_thresh)
        feature_dict[col_prefix + 'Number of Peaks'] = len(peaks[0])

    feature_dict[col_prefix + 'Percentage Fractionation'] = percentage_fractionation(x, peaks[0])

    # Denoise x and see if performance is affected
    x = denoise(x)

    # Abritrary selection of features from TSFresh
    erbc = feature_calculators.energy_ratio_by_chunks(x, [{'num_segments':10, 'segment_focus':3}, {'num_segments':10, 'segment_focus':2}])
    feature_dict[col_prefix + 'Energy Ratio by Chunks: num_segments=10 segment_focus=2'] = erbc[1][1]
    feature_dict[col_prefix + 'Approximate Entropy: m=2 r=0.7'] = feature_calculators.approximate_entropy(x, 2, 0.7)
    # feature_dict[col_prefix + 'Ratio Beyond 5xSTD'] = feature_calculators.ratio_beyond_r_sigma(x, 5)
    # A fraction q of the mass lies to the left of i. (Alternative to conduction delay?)
    imq = feature_calculators.index_mass_quantile(x, [{'q': 0.6}, {'q': 0.4}])
    # feature_dict[col_prefix + 'Index Mass Quantile: q=0.6'] = imq[0][1]
    # feature_dict[col_prefix + 'Kurtosis'] = feature_calculators.kurtosis(x)
    # feature_dict[col_prefix + 'Binned Entropy: bins=10'] = feature_calculators.binned_entropy(x, 10)
    # feature_dict[col_prefix + 'Binned Entropy: bins=20'] = feature_calculators.binned_entropy(x, 20)
    # feature_dict[col_prefix + 'Quantile: 0.9'] = feature_calculators.quantile(x,0.9)
    # ML AR process fit.
    # ar_coeffs = feature_calculators.ar_coefficient(x, [{'coeff': 2, 'k': 10}])
    # feature_dict[col_prefix + 'AR Coefficient: coeff=2 k=10'] = ar_coeffs[0][1]
    # feature_dict[col_prefix + 'Standard Deviation'] = np.std(x)
    # feature_dict[col_prefix + 'Normalised Standard Deviation'] = np.std(x/max(abs(x)))

    return feature_dict

# A shitty conduction delay detector
def get_delay(x, amp_thresh=None, set_thresh=False):
    if (set_thresh==True):
        if any(abs(x)>amp_thresh):
            return np.argmax(abs(x)>amp_thresh)
        else:
            return len(x)
    else:
        return np.argmax(abs(x)>(max(abs(x))/4))

def percentage_fractionation(x, peak_idxs, sr=1000):
    # Get peak indexes and amplitude
    peak_idx_diffs = np.diff(peak_idxs)
    frac_time = 0
    frac_time = np.sum(peak_idx_diffs[peak_idx_diffs < 0.01*sr])
    prcnt_frac = (frac_time/len(x))*100
    return prcnt_frac

def sample_entropy(U, m, r):

    def _maxdist(x_i, x_j):
        result = max([abs(ua-va) for ua, va in zip(x_i, x_j)])
        return result

    def _phi(m):
        x = np.zeros([N,m-1])
        for i in range(N-m+1):
            x[i,:] = U[i:i+m-1]

        C = 0
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    if _maxdist(x[i,:], x[j,:]) <= r:
                        C = C + 1

        return C

    U = U/max(abs(U))
    N = len(U)

    return -np.log(_phi(m+1)/_phi(m))
