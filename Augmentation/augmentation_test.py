import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import parser
import numpy as np
import matplotlib.pyplot as plt
import percentage_fractionation
import data_augmentation
from FeatureExtraction import get_segment, feature_tools

fileLoc = '/Users/matthewashman/github/MasterProject2018/Data/dataset00/'
augmentedData = np.zeros([25,150])
originalData = np.zeros([25,150])
originalFeatures = np.zeros([25,2])
augmentedFeatures = np.zeros([25,2])
for i, patient in enumerate(['e', 'f', 'g', 'h', 'i']):
    for j, fileNum in enumerate(['1', '2', '3', '4', '5']):
        dataIdx = i*5 + j
        file = patient + '0' + fileNum
        fileName = fileLoc + 'TESTEXPORT' + file + '.txt'
        data, sr, numSamples = parser.parseFile(fileName)

        # data.plot(subplots=True, sharex=True)
        # plt.suptitle(fileLoc + ": " + file)
        # plt.xlabel('Sample')
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1)
        # plt.draw()
        # plt.waitforbuttonpress()
        # plt.close()

        v = data['V1']
        try:
            x = data['CSd']
        except KeyError as e:
            try:
                x = data['CS1-2']
            except KeyError as e:
                x = data['CS 1-2']

        try:
            s = data['CSp']
        except KeyError as e:
            try:
                s = data['CS9-10']
            except KeyError as e:
                s = data['CS 9-10']


        try:
            seg = get_segment.get_segment(x, s, v, sr, False)
            seg.reset_index(drop=True, inplace=True)
            originalData[dataIdx,:] = seg
            originalFeatures[dataIdx,0] = percentage_fractionation.percentage_fractionation(seg, sr)
            originalFeatures[dataIdx,1] = sample_entropy.sample_entropy(seg, 2, 0.05)
            segAug = data_augmentation.augment_fractionation(seg, True)
            augmentedData[dataIdx,:] = segAug
            augmentedFeatures[dataIdx,0] = percentage_fractionation.percentage_fractionation(segAug, sr)
            augmentedFeatures[dataIdx,1] = sample_entropy.sample_entropy(segAug, 2, 0.05)
        except:
            seg = np.zeros(150)
            segAug = np.zeros(150)
            originalData[dataIdx,:] = seg
            augmentedData[dataIdx,:] = segAug
            originalFeatures[dataIdx,0] = percentage_fractionation.percentage_fractionation(seg, sr)
            originalFeatures[dataIdx,1] = sample_entropy.sample_entropy(seg, 2, 0.01)
            augmentedFeatures[dataIdx,0] = percentage_fractionation.percentage_fractionation(segAug, sr)
            augmentedFeatures[dataIdx,1] = sample_entropy.sample_entropy(segAug, 2, 0.01)

print(augmentedFeatures)
print(originalFeatures)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(originalFeatures[:,0], originalFeatures[:,1], s=10, c='b', label='Original Data')
ax1.scatter(augmentedFeatures[:,0], augmentedFeatures[:,1], s=10, c='r', label='Augmented Data')
plt.legend(loc='upper left');
plt.draw()
plt.waitforbuttonpress()
plt.close()


# fileLoc = '/Users/matthewashman/github/MasterProject2018/Data/dataset01/'
# files = ['0270', '0280', '0290', '0300', '0310', '0320', '0330', '0340', '0350', '0360', '0370', '0380']
# features = np.zeros([len(files),2])
# for fileIdx in range(len(files)):
#     file = files[fileIdx]
#     fileName = fileLoc + 'TESTEXPORT' + file + '.txt'
#     data, sr, numSamples = parser.parseFile(fileName)
#
#     # data.plot(subplots=True, sharex=True)
#     # plt.suptitle(fileLoc + ": " + file)
#     # plt.xlabel('Sample')
#     # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1)
#     # plt.draw()
#     # plt.waitforbuttonpress()
#     # plt.close()
#
#     v = data['V1']
#     s = data['H 7-8']
#     x = data['H 1-2']
#
#     seg = get_segment.get_segment(x, s, v, sr, False)
#     features[fileIdx,0] = percentage_fractionation.percentage_fractionation(seg, sr)
#     features[fileIdx,1] = sample_entropy.sample_entropy(seg, 2, 0.05)
#
# print(features)
# fig = plt.figure()
# plt.scatter(features[:,0], features[:,1])
# for i, txt in enumerate(files):
#     plt.annotate(txt, (features[i,0], features[i,1]))
# plt.xlabel('Percentage Fractionation')
# plt.ylabel('Sample Entropy')
# plt.draw()
# plt.waitforbuttonpress()
# plt.close(fig)
