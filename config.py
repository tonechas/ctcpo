"""Settings for texture classification through partial orders.

This module defines the following global variables:

    home : str
        Path of the directory where all data are located.
    imgs : str
        Path of the directory where the images are located.
    data : str 
        Path of the directory where features and classification results 
        are saved.
    log : str
        Path of the directory where log files are stored.
    estimators : list of tuples
        Each tuple is formed by a classifier and a dictionary with the 
        set of values of the hyper-parameters over which the grid search 
        is going to be performed.
    n_tests : int
        Number of random splits of the image dataset into train and test sets.
    n_folds : int
        Number of folds used for cross-validation.
    test_size : float
        Proportion of samples used for testing. Expressed as per-unit.
"""


import os
import platform
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


if platform.system() == 'Linux':
    home = r'/mnt/netapp2/Store_uni/home/uvi/dg/afa/texture'
elif platform.system() == 'Windows':
    home = r'C:\texture'


imgs = os.path.join(home, 'images')
data = os.path.join(home, 'data')
log = os.path.join(home, 'log')


estimators = [
    (KNeighborsClassifier, dict(n_neighbors=[1, 3, 5, 7, 9, 11])),
#    (SVC, dict(C=np.logspace(-3, 2, 6), gamma=np.logspace(-3, 2, 6))),
    ]

n_tests = 5
n_folds = 5
test_size = 1/2
random_state = 0
