import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from epdata_tools import epdata_main, get_ep_features
from IPython.display import HTML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, naive_bayes, neighbors, gaussian_process
from sklearn.gaussian_process.kernels import RBF
import xgboost

import pdb
import time

plt.style.use('default')

# N_aug = 5   # Number of augmented examples per file.
file_loc = '/Users/matthewashman/github/MasterProject2018/Data/EPdata/'
# Patient 7 is awful
af_patients = ('1', '2', '3', '4', '5', '6', '8', '9', '10') # AF patients
at_patients = ('1', '2', '3') # AT patients
avnrt_patients = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22') # AVNRT patients
avrt_patients = ('1', '2', '3', '4', '5', '6', '7', '8') # AVRT patients
ep_patients = ('1', '2', '3', '4', '5', '6', '7', '8') # EP patients
coupling_intervals = ('220','230', '240', '250', '260', '270', '280', '290', '300',
                      '310', '320', '330', '340', '350', '360', '370', '380', '390', '400')

X_af = epdata_main(file_loc, af_patients, 'af', tuple(reversed(coupling_intervals)))

cs12_labels = ( ('250', '240'),
                ('260', '250'),
                ('350', '340', '330', '320', '310', '300', '290', '280', '270', '260', '250'),
                ('250'),
                ('300', '290', '280', '270', '260', '250', '240'),
                ('320', '290', '280'),
                (),
                ('340', '330', '320', '310', '300', '290', '280', '270', '260', '250'),
                ('320', '310', '300', '290', '280', '270', '260', '250'),
                ('320', '300', '290', '280', '270', '260', '250')
                )

cs34_labels = ( ('260, 250, 240'),
                ('290', '280', '270', '260', '250'),
                ('280', '270', '260'),
                ('260', '250'),
                ('270', '260', '250', '240'),
                ('310', '300', '290', '280'),
                ('340', '330', '320', '310', '300', '290'),
                ('280', '270', '260', '250'),
                ('280', '270', '260'),
                ('300', '290', '280', '270', '260', '250')
                )
X_af['Label'] = np.nan

# X_12 and X_34 will store feature vectors. y_12 and y_34 will store labels. idx_12 and idx_34 will store index in X_af.
X_12 = []; X_34 = []; y_12 = []; y_34 = []; idx_12 = []; idx_34 = []

for patient_idx, patient in enumerate(af_patients):
    print('Evaluating patient: ' + patient)
    # Add labels first
    X_af.loc[(X_af['Patient']==patient) & (X_af['S1/S2']=='S2'), 'Label'] = 0
    for ci in cs12_labels[patient_idx]:
        X_af.loc[(X_af['Patient']==patient) & (X_af['Channel']=='CS1-2') & (X_af['Coupling Interval']==ci) & (X_af['S1/S2']=='S2'), 'Label'] = 1

    for ci in cs34_labels[patient_idx]:
        X_af.loc[(X_af['Patient']==patient) & (X_af['Channel']=='CS3-4') & (X_af['Coupling Interval']==ci) & (X_af['S1/S2']=='S2'), 'Label'] = 1

    # Group segments for now. Preserving indices will come in handy later on.
    patient_data = X_af[(X_af['Type']=='af') & (X_af['Patient']==patient)]
    cs12_s2_data = patient_data[(patient_data['Channel']=='CS1-2') & (patient_data['S1/S2']=='S2')]
    cs12_s1_data = patient_data[(patient_data['Channel']=='CS1-2') & (patient_data['S1/S2']=='S1')]
    cs34_s2_data = patient_data[(patient_data['Channel']=='CS3-4') & (patient_data['S1/S2']=='S2')]
    cs34_s1_data = patient_data[(patient_data['Channel']=='CS3-4') & (patient_data['S1/S2']=='S1')]

    # Then add feature vectors
    # Get typical S1 response for DTW calculation
    typical_cs12_s1 = cs12_s1_data.iloc[0]['Data']
    typical_cs12_s1_feature_vec = get_ep_features(typical_cs12_s1, typical_cs12_s1)
    for idx,s2 in cs12_s2_data.iterrows():
        # Get other features
        feature_vec = get_ep_features(s2['Data'], typical_cs12_s1)
        feature_vec = feature_vec - typical_cs12_s1_feature_vec
        X_12.append(feature_vec)
        y_12.append(s2['Label'])
        idx_12.append(idx)

    typical_cs34_s1 = cs34_s1_data.iloc[0]['Data']
    typical_cs34_s1_feature_vec = get_ep_features(typical_cs34_s1, typical_cs34_s1)
    for idx, s2 in cs34_s2_data.iterrows():
        # Get other features
        feature_vec = get_ep_features(s2['Data'], typical_cs34_s1)
        feature_vec = feature_vec - typical_cs34_s1_feature_vec
        X_34.append(feature_vec)
        y_34.append(s2['Label'])
        idx_34.append(idx)

X_12 = np.asarray(X_12)
y_12 = np.asarray(y_12)
idx_12 = np.asarray(idx_12)
X_34 = np.asarray(X_34)
y_34 = np.asarray(y_34)
idx_34 = np.asarray(idx_34)

X = np.concatenate((X_12,X_34),axis=1)
y = np.logical_or(y_12, y_34).astype(int)
idx = np.concatenate((idx_12.reshape(-1,1), idx_34.reshape(-1,1)),axis=1)

X_12_train, X_12_test, y_12_train, y_12_test, idx_12_train, idx_12_test = train_test_split(X_12, y_12, idx_12, test_size=0.3)
X_34_train, X_34_test, y_34_train, y_34_test, idx_12_train, idx_12_test = train_test_split(X_34, y_34, idx_34, test_size=0.3)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, idx, test_size=0.3)

models = (svm.SVC(), naive_bayes.GaussianNB(), neighbors.KNeighborsClassifier(),
        gaussian_process.GaussianProcessClassifier(kernel=1.0*RBF(1)), xgboost.XGBClassifier())
models = (clf.fit(X_train,y_train) for clf in models)
for clf in models:
    print(cross_val_score(clf, X_train, y_train, cv=3))

clf = gaussian_process.GaussianProcessClassifier(kernel=1.0*RBF(1))
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
predictions = clf.predict(X_test)
mistake_idxs = idx_test[predictions != y_test]
mistake_labels = np.concatenate((predictions[predictions != y_test].reshape(-1,1), y_test[predictions != y_test].reshape(-1,1)), axis=1)

mistake_idxs = np.squeeze(mistake_idxs)

print(mistake_labels)

for i, [cs12_idx, cs34_idx] in enumerate(mistake_idxs):
    patient = X_af['Patient'].loc[cs12_idx]
    coupling_interval = X_af['Coupling Interval'].loc[cs12_idx]
    print(X_af['Coupling Interval'].loc[cs34_idx])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    [ax1, ax2] = axes.flatten()
    ax1.plot(X_af['Data'].loc[cs12_idx])
    ax1.set_title('CS1-2')
    ax2.plot(X_af['Data'].loc[cs34_idx])
    ax2.set_title('CS3-4')
    plt.suptitle('Patient: ' + patient + ' Coupling Interval: ' + coupling_interval + '\n Predicted label: ' + str(mistake_labels[i,0]) + ' True label: ' + str(mistake_labels[i,1]))
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
