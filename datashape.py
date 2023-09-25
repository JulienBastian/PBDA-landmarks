import argparse
from functools import partial
import multiprocessing
from multiprocessing import Pool

import time 

import os
from os.path import join, abspath, dirname, exists
from os import makedirs

import pickle
import numpy as np

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold

from sklearn.datasets import make_moons

from pbrff.data_loader import DataLoader
from pbrff.baseline import learn_svm, learn_TSVM, learn_dalc, learn_pbda
from pbrff.greedy_kernel import GreedyKernelLearner, compute_greedy_kernel
from pbrff.landmarks_based import compute_landmarks_selection, compute_landmarks_based, compute_landmarks_based_DA

import warnings
warnings.filterwarnings("ignore")

from dataset import *

for z in range(1):
    source_files=["books.dvd_source.svmlight",
    "books.kitchen_source.svmlight"
    ]

    target_files=["books.dvd_target.svmlight",
    "books.kitchen_target.svmlight"
   ]

    test_files=["books.dvd_test.svmlight",
    "books.kitchen_test.svmlight"
    ]

    
    dataset_list=["books.dvd",
    "books.kitchen"
    ]
    ######


    # Loading dataset
    #dataloader = DataLoader(random_state=random_state)
    #X_source, X_target, y_source, y_target = dataloader.load(args.dataset)


    ####Datasets

    ##Génération d'un source 

    #X_S_test, y_S_test, nul, nul= make_moons_da(6000, rotation=00, noise=0.1, random_state=random_state)
    
    #X_source, X_trash, y_source, y_trash = train_test_split(X_source, y_source, test_size=0.8, random_state=random_state)
    #X_target, X_trash, y_target, y_trash = train_test_split(X_target, y_target, test_size=0.8, random_state=random_state)

    #degrees=[10,20,30,40,50,70, 90]
    deg=["real"]
    iFoldVal=0
    n_runs=1
    nbFoldValid=5
    for i in range(len(source_files)):
        print("real experiment on" + str(dataset_list[i]))

        source_data = dataset_from_svmlight_file(source_files[i])
        target_data = dataset_from_svmlight_file(target_files[i], source_data.get_nb_features())
        test_data = dataset_from_svmlight_file(test_files[i], target_data.get_nb_features())
        source_data.reshape_features(target_data.get_nb_features())
        X_source=source_data.X
        y_source=source_data.Y.astype(int)
        print(X_source.shape)
        X_target=target_data.X
        y_target=target_data.Y.astype(int)
        print(X_target.shape)
        X_T_test=test_data.X
        y_T_test=test_data.Y.astype(int)
        print(X_T_test.shape)