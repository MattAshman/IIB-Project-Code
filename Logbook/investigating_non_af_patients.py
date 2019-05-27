# Import necessary modules. Set settings. Import data.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from epdata_tools import epdata_main, get_ep_features, get_ep_feature_dict
from IPython.display import HTML

from Augmentation import data_augmentation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn import svm, naive_bayes, neighbors, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF

import xgboost

from IPython.display import display, clear_output
import pdb

plt.style.use('default')

X_compact = pd.read_pickle('/Users/matthewashman/github/MasterProject2018/Data/X_all_compact.pkl')

patient_types = list(X_compact['Type'].unique())
# patient_types.remove('af')

for patient_type in patient_types:
    patients = X_compact[X_compact['Type']==patient_type]['Patient'].unique()
    for patient in patients:
        patient_X = X_compact[(X_compact['Type']==patient_type) & (X_compact['Patient']==patient)]

        s1 = patient_X[patient_X['S1/S2']=='S1'].sort_values(by=['Coupling Interval'], ascending=False)
        s1 = s1.reset_index()

        s2 = patient_X[patient_X['S1/S2']=='S2'].sort_values(by=['Coupling Interval'], ascending=False)
        s2 = s2.reset_index()


        # Plot patient data as coupling interval decreases
        fig, axes = plt.subplots(nrows=s1.shape[0], ncols=6, sharex=True, figsize=(16,9))

        # Plot S1 data
        for i, row in s1.iterrows():
            # Plot CS1-2 data
            axes[i][0].plot(row['CS1-2'])
            axes[i][0].set_ylabel(row['Coupling Interval'])
            # Plot CS3-4 data
            axes[i][2].plot(row['CS3-4'])
            axes[i][2].set_ylabel(row['Coupling Interval'])
            # Plot CS5-6 data
            axes[i][4].plot(row['CS5-6'])
            axes[i][4].set_ylabel(row['Coupling Interval'])

        # Plot S2 data
        for i, row in s2.iterrows():
            # Plot CS1-2 data
            axes[i][1].plot(row['CS1-2'])
            axes[i][1].set_ylabel(row['Coupling Interval'])
            # Plot CS3-4 data
            axes[i][3].plot(row['CS3-4'])
            axes[i][3].set_ylabel(row['Coupling Interval'])
            # Plot CS5-6 data
            axes[i][5].plot(row['CS5-6'])
            axes[i][5].set_ylabel(row['Coupling Interval'])

        axes[0][0].set_title('CS1-2 S1')
        axes[0][1].set_title('CS1-2 S2')
        axes[0][2].set_title('CS3-4 S1')
        axes[0][3].set_title('CS3-4 S2')
        axes[0][4].set_title('CS5-6 S1')
        axes[0][5].set_title('CS5-6 S2')

        for ax in axes.ravel():
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off

            ax.tick_params(
                            axis='y',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            left=False,      # ticks along the bottom edge are off
                            right=False,         # ticks along the top edge are off
                            labelleft=False) # labels along the bottom edge are off

        plt.suptitle('S1/S2 responses in channels CS1-2, CS3-4 and CS5-6 for patient ' + patient_type + patient)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
