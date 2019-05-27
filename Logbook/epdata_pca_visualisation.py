import sys
sys.path.insert(0, '/Users/matthewashman/github/MasterProject2018')

import copy
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parser
from FeatureExtraction import get_segment, feature_tools
from Augmentation import data_augmentation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text

plt.style.use('default')

N_aug = 5   # Number of augmented examples per file.
file_loc = '/Users/matthewashman/github/MasterProject2018/Data/dataset01/'
files = ['0270', '0280', '0290', '0300', '0310', '0320', '0330', '0340', '0350', '0360', '0370', '0380']
features = np.zeros([len(files),5])
augmented_features = np.zeros([N_aug*len(files),5])
for file_idx in range(len(files)):
    # Extract data from file
    file = files[file_idx]
    file_name = file_loc + 'TESTEXPORT' + file + '.txt'
    data, sr, num_samples = parser.parseFile(file_name)

    # Use the following channels
    v = data['V1']
    s = data['H 7-8']
    x = data['H 1-2']

    # Get segment
    seg = get_segment.get_segment(x, s, v, sr, False)
    seg = seg.values
    # Normalise segment
    seg = seg/max(abs(seg))

    seg_aug = data_augmentation.augment_transformation(seg, N_aug, False)

    features[file_idx,:3] = feature_tools.basic_features(seg, sr)
    features[file_idx,3] = feature_tools.sample_entropy(seg, 2, 0.05)
    features[file_idx,4] = feature_tools.percentage_fractionation(seg, sr)

    for i in range(N_aug):
        augmented_features[N_aug*file_idx-(N_aug-i),:3] = feature_tools.basic_features(seg_aug[i,:], sr, False)
        augmented_features[N_aug*file_idx-(N_aug-i),3] = feature_tools.sample_entropy(seg_aug[i,:], 2, 0.05)
        augmented_features[N_aug*file_idx-(N_aug-i),4] = feature_tools.percentage_fractionation(seg_aug[i,:], sr)

# Normalise feature vectors
scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)
combined_features = np.concatenate((features, augmented_features), axis=0)
scaled_features = scaler.fit_transform(combined_features)

# Perform 2D PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
pca_components = pca.components_
# pca_augmented_features = np.matmul(augmented_features, np.transpose(pca_components))
plt.figure(figsize=(6,5))
ax = plt.gca()
plt.grid(True)
plt.scatter(pca_features[:12,0], pca_features[:12,1], alpha=0.75)
# plt.hold(True)
# plt.scatter(pca_augmented_features[:,0], pca_augmented_features[:,1])

texts = [plt.text(pca_features[i,0], pca_features[i,1], '%s' %file[1:], ha='center', va='center') for i, file in enumerate(files)]
adjust_text(texts)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.draw()
plt.waitforbuttonpress()
plt.close()

plt.figure(figsize=(6,5))
ax = plt.gca()
plt.grid(True)
# Colour 270, 310, 360 and show augmented data.
pca_features_270 = np.vstack((pca_features[0,:], pca_features[12:17,:]))
pca_features_310 = np.vstack((pca_features[4,:], pca_features[32:37,:]))
pca_features_360 = np.vstack((pca_features[9,:], pca_features[57:62,:]))
plt.scatter(pca_features_270[0,0], pca_features_270[0,1], c='r')
plt.text(pca_features_270[0,0], pca_features_270[0,1]+0.05, '270', ha='center')
plt.hold(True)
plt.scatter(pca_features_270[1:,0], pca_features_270[1:,1], c='r', alpha=0.5)
plt.hold(True)
plt.scatter(pca_features_310[0,0], pca_features_310[0,1], c='b')
plt.text(pca_features_310[0,0], pca_features_310[0,1]+0.05, '310', ha='center')
plt.hold(True)
plt.scatter(pca_features_310[1:,0], pca_features_310[1:,1], c='b', alpha=0.5)
plt.hold(True)
plt.scatter(pca_features_360[0,0], pca_features_360[0,1], c='g')
plt.text(pca_features_360[0,0], pca_features_360[0,1]+0.05, '360', ha='center')
plt.hold(True)
plt.scatter(pca_features_360[1:,0], pca_features_360[1:,1], c='g', alpha=0.5)
# plt.hold(True)
# plt.scatter(pca_augmented_features[:,0], pca_augmented_features[:,1])

# texts = [plt.text(pca_features[i,0], pca_features[i,1], '%s' %file[1:], ha='center', va='center') for i, file in enumerate(files)]
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.draw()
plt.waitforbuttonpress()
plt.close()
