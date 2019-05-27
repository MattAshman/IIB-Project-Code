import numpy as np
import math
import pywt
import peakutils
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import entr
from statsmodels.robust import mad
import detect_peaks

def showSegments(x, segments, segmentIdxs):
    plt.figure()
    plt.plot(x, 'b')
    plt.hold(True)
    for i, segment in enumerate(segments):
        xAxis = np.arange(segmentIdxs[i][0], segmentIdxs[i][1])
        plt.plot(xAxis, segment, 'r')
    plt.hold(False)
    plt.show()

def denoise(x):
    """
    Inputs
    x: 1D array to be xDenoised

    Outputs
    xDenoised: Denoised x

    Denoising is performed by wavelet decomposition. Daubechies N=8 wavelet
    coefficients are first obtained, then coefficients below a adaptive
    threshold are discarded. The inverse DWT is then applied, and the denoised
    input is returned.
    """
    # Obtain Daubechies N=8 wavelet coefficients
    waveletCoefs = pywt.wavedec(x, 'db6', mode='per')

    # Throw away coefficients corresponding to noise
    sigma = mad(waveletCoefs[-1])
    uThresh = sigma*np.sqrt(2*np.log(len(x)))
    denoised = waveletCoefs[:]
    denoised[1:] = (pywt._thresholding.soft(i, value=uThresh) for i in denoised[1:])

    # Reconstruct the original signal
    xDenoised = pywt.waverec(denoised, 'db6', mode='per')

    return xDenoised

def getSegments(x, ampThresh, segLength):
    """
    Inputs
    x: 1D array from which segments are obtained.
    ampThresh: Segment begins when x exceeds ampThresh.
    segLength: Length of segments.

    Outputs
    segments: List of segments obtained from x.
    segmentIDxs: List of start/end indeces for segments.

    Segments are defined as fixed length (segLength) segments of x that begin
    when the absolute amplitude of x exceeds ampThresh.
    """
    segments = list()
    segmentIdxs = list()
    idx = 0
    while(idx<len(x)):
        idx = idx + np.idxmax(abs(x[idx:])>ampThresh)
        if ~((idx+segLength)>len(x)):
            segments.append(x[(idx-1):(idx+segLength)])
            segmentIdxs.append([(idx-1), (idx+segLength)])
        idx = idx + segLength

    return segments, segmentIdxs

def getPeakFeatures(x, heightThresh, fileName = None, plot = False):
    """
    Inputs
    x: 1D array from which peak features are extracted
    heighTresh: Minimum height of peak.

    Outputs
    peakIdx: Indeces of peaks.
    peakAmp: Amplitudes of respective peaks.

    Finds the number of peaks in x with amplitude greater than heightThresh.
    """
    x = np.array(x)
    denoisedX = denoise(x)
    sf = max(abs(x))
    denoisedSf = max(abs(denoisedX))
    x = x/sf
    denoisedX = denoisedX/denoisedSf
    maxPos = max(x)
    maxNeg = max(-x)
    neighbourThresh = 0.01
    posPeakIdx = detect_peaks.detect_peaks(denoisedX, mph=heightThresh, threshold = 0.005)
    negPeakIdx = detect_peaks.detect_peaks((-denoisedX), mph=heightThresh, threshold = 0.005)
    # posPeakIdx = detect_peaks.detect_peaks(x, mph=heightThresh, threshold = 0.01)
    # negPeakIdx = detect_peaks.detect_peaks((-x), mph=heightThresh, threshold = 0.01)
    # smallPosPeakIdx = detect_peaks.detect_peaks(x, mph=heightThresh, maxph = 4*heightThresh, threshold = 0.02)
    # smallNegPeakIdx = detect_peaks.detect_peaks((-x), mph=heightThresh, maxph = 4*heightThresh, threshold = 0.02)
    # medPosPeakIdx = detect_peaks.detect_peaks(x, mph=4*heightThresh, threshold = 0.005)
    # medNegPeakIdx = detect_peaks.detect_peaks((-x), mph=4*heightThresh, threshold = 0.005)
    # largePosPeakIdx = detect_peaks.detect_peaks(x, mph=maxPos)
    # largeNegPeakIdx = detect_peaks.detect_peaks((-x), mph=maxNeg)
    # posPeakIdx = peakutils.indexes(x, thres = 0.75)
    # negPeakIdx = peakutils.indexes((-x), thres = 0.75)
    # posPeakIdx,_ = signal.find_peaks((x), height = heightThresh)
    # negPeakIdx,_ = signal.find_peaks((-x), height = heightThresh)
    # posPeakIdx = signal.find_peaks_cwt(x, np.arange(5,10))
    # negPeakIdx = signal.find_peaks_cwt((-x), np.arange(5,10))
    # peakIdx = np.concatenate([smallPosPeakIdx, largePosPeakIdx, medPosPeakIdx, medNegPeakIdx, smallNegPeakIdx, largeNegPeakIdx])
    peakIdx = np.concatenate([posPeakIdx, negPeakIdx])
    peakIdx = np.sort(peakIdx)
    peakAmp = x[peakIdx]
    newPeakIdx = []
    newPeakAmp = []
    if (len(peakIdx) > 0):
        # Remove multiple peaks due to clipping
        newPeakIdx.append(peakIdx[0])
        newPeakAmp.append(peakAmp[0])
        for i in range(1, len(peakIdx)):
            if not (abs(peakAmp[i]-peakAmp[i-1]) < 0.02):
                newPeakIdx.append(peakIdx[i])
                newPeakAmp.append(peakAmp[i])

    # Convert back to np array
    peakIdx = np.array(newPeakIdx)
    peakAmp = np.array(newPeakAmp)*denoisedSf

    if plot == True:    
        fig = plt.figure()
        plt.plot(x, 'b' , denoisedX, 'r--', peakIdx, peakAmp/denoisedSf, 'kx')
        plt.title(fileName)
        plt.xlabel('Sample')
        plt.ylabel('Normalised amplitude')
        plt.legend(['Original segment', 'Denoised segment', 'Detected peaks'])
        plt.draw()
        plt.waitforbuttonpress(0) # this will wait for indefinite time
        plt.close(fig)

    return peakIdx, peakAmp

def getShannonEntropy(x, numBins):
    maxVal = 1
    minVal = -1
    binWidth = (maxVal-minVal)/numBins

    # Assume minVal is negative
    binCentres = np.arange((minVal+binWidth/2), (maxVal+binWidth/2), binWidth)

    binCount = np.zeros(len(binCentres))
    xQuantised = np.zeros(len(x))
    for i, binCentre in enumerate(binCentres):
        idxs = (x > (binCentre-binWidth/2)) & (x < (binCentre+binWidth/2))
        xQuantised[idxs] = binCentre

        # Store number of values quantised to binCentre
        binCount[i] = np.sum(idxs)

    # Probabilities must sum to 1
    probabilities = binCount/np.sum(binCount)
    # Remove 0 probabilities
    probabilities = probabilities[probabilities != 0]
    entropy = 0

    # Calculate Shannon entropy
    for p in probabilities:
        entropy = entropy - p*(math.log(p, 2))

    return entropy
