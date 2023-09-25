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

##Python file to run the experiments on real data
from dataset import *

RESULTS_PATH = os.environ.get('PBRFF_RESULTS_DIR', join(dirname(abspath(__file__)), "results"))

def main():
    parser = argparse.ArgumentParser(description="PAC-Bayes RFF Experiment")
    parser.add_argument('-dsource', '--datasetsource', type=str, default="books.dvd_source.svmlight")
    parser.add_argument('-dtarget', '--datasettarget', type=str, default="books.dvd_target.svmlight")
    parser.add_argument('-dtest', '--datasettest', type=str, default="books.dvd_test.svmlight")
    parser.add_argument('-e', '--experiments', type=str, nargs='+', default=["landmarks_based"])
    parser.add_argument('-l', '--landmarks-method', type=str, nargs='+', default=["random"])
    parser.add_argument('-n', '--n-cpu', type=int, default=-1)
    parser.add_argument('-r', '--reversevalidation', type=str, default="no")
    args = parser.parse_args()

    reversevalidation=args.reversevalidation
    # Setting random seed for repeatability
    random_seed = 1062023
    #random_state = check_random_state(random_seed)
    random_state = random_seed
    # Number of CPU for parallel computing
    if args.n_cpu == -1:
        n_cpu = multiprocessing.cpu_count()
    else:
        n_cpu = args.n_cpu
    print(f"Running on {n_cpu} cpus")

    # Preparing output paths
    paths = {'cache': join(RESULTS_PATH, "cache", args.dataset),
             'baseline': join(RESULTS_PATH, "baseline", args.dataset),
             'greedy_kernel': join(RESULTS_PATH, "greedy_kernel", args.dataset)}
    paths.update({f'landmarks_based_{l}':  join(RESULTS_PATH, "landmarks_based", l, args.dataset) for l in args.landmarks_method})

    for path_name, path in paths.items():
        if (not exists(path)): makedirs(path)

    deg=["real"]
    iFoldVal=5
    n_runs=10
    nbFoldValid=5

    print("real experiment on")

    source_data = dataset_from_svmlight_file(source_files[i])
    target_data = dataset_from_svmlight_file(target_files[i], source_data.get_nb_features())
    test_data = dataset_from_svmlight_file(test_files[i], target_data.get_nb_features())
    source_data.reshape_features(target_data.get_nb_features())
    X_source=source_data.X
    y_source=source_data.Y.astype(int)
    X_target=target_data.X
    y_target=target_data.Y.astype(int)
    X_T_test=test_data.X
    y_T_test=test_data.Y.astype(int)

    for run in range(n_runs):
        X_S_train, X_S_valid, y_S_train, y_S_valid = train_test_split(X_source, y_source, test_size=0.2, random_state=random_state)
        X_T_train, X_T_valid, y_T_train, y_T_valid = train_test_split(X_target, y_target, test_size=0.2, random_state=int(random_state*((run+1)*2)))
        for iFoldVal in [1]:
                start_time=time.time()
                dataset = {'name': args.datasetsource ,
                               'X_S_train': X_S_train, 'X_T_train': X_T_train, 'X_S_valid': X_S_valid, 'X_T_test': X_T_test,'X_S_test': X_S_valid, 'X_T_valid': X_T_valid,
                               'y_S_train': y_S_train, 'y_T_train': y_T_train, 'y_S_valid': y_S_valid, 'y_T_test': y_T_test, 'y_S_test': y_S_valid,'y_T_valid': y_T_valid}
                hps = {     'gamma' : [0.1],
                            'C': [1],
                            'beta': np.logspace(-3, 3, 3),
                            #'beta_da': [10.0, 1.0, 0.01],
                            'beta_da': [1.0],
                            'c':[10.0, 1.0, 0.01],
                            'b':[10.0, 1.0, 0.01],
                            'C_dalc' : [10.0, 1.0, 0.01],
                            'B_dalc' : [10.0, 1.0, 0.01],
                            'C_pbda' : [10.0, 1.0, 0.01],
                            'A_pbda' : [10.0, 1.0, 0.01],
                            'landmarks_percentage': [0.01, 0.05, 0.1, 0.15, 0.20, 0.25],
                            'nb_landmarks':np.logspace(1, 2, 2).astype(int),
                            'landmarks_D': np.logspace(1, 2, 2).astype(int),
                            'rho': [1.0, 0.1, 0.01, 0.001, 0.0001],
                            'greedy_kernel_N': 20000,
                            'greedy_kernel_D': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275,\
                               300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500, 1750, 2000, 2500, 3000,\
                               3500, 4000, 4500, 5000]}
                    
                svm_file = join(paths['baseline'], f"svm_dataset_{args.datasetsource}.pkl")
                print(f"run {run+1}")
                learn_svm(dataset=dataset,
                                C_range=hps['C'],
                                gamma_range=hps['gamma'],
                                output_file=svm_file,
                                n_cpu=n_cpu,
                                k=iFoldVal+1,
                                nrun=run+1,
                                deg=deg,
                                random_state=int(random_state*(run+1)*(iFoldVal+1)))

                with open(svm_file, 'rb') as in_file:
                    svm_results = pickle.load(in_file)

                    
                    
                tsvm_file = join(paths['baseline'], f"tsvm_dataset_{args.datasetsource}.pkl")

                learn_TSVM(dataset=dataset,
                                C_range=hps['C'],
                                gamma_range=hps['gamma'],
                                output_file=tsvm_file,
                                n_cpu=n_cpu,
                                k=iFoldVal+1,
                                nrun=run+1,
                                deg=deg,
                                random_state=int(random_state*(run+1)*(iFoldVal+1)),
                                reverse="no")
                    
                with open(tsvm_file, 'rb') as in_file:
                    tsvm_results = pickle.load(in_file)
                gamma = tsvm_results[0]["gamma"]
                    
                dalc_file = join(paths['baseline'], f"dalc_dataset_{args.datasetsource}.pkl")

                learn_dalc(dataset=args.datasetsource,
                                C_dalc_range=hps['C_dalc'],
                                B_dalc_range=hps['B_dalc'],
                                output_file=dalc_file,
                                k=iFoldVal+1,
                                nrun=run+1,
                                deg=deg,
                                random_state=int(random_state*(run+1)*(iFoldVal+1)),
                                reverse="yes")
                    
                pbda_file = join(paths['baseline'], f"pbda_dataset_{args.datasetsource}.pkl")

                learn_pbda(dataset=dataset,
                                C_pbda_range=hps['C_pbda'],
                                A_pbda_range=hps['A_pbda'],
                                output_file=pbda_file,
                                k=iFoldVal+1,
                                nrun=run+1,
                                deg=deg,
                                random_state=int(random_state*(run+1)*(iFoldVal+1)),
                                reverse="yes")

                    


                if "landmarks_based" in args.experiments:

                    # Initializing landmarks-based learners by selecting landmarks according to methods
                    param_grid = ParameterGrid([{'method': args.landmarks_method, 'nb_landmarks': hps['nb_landmarks']}])
                    param_grid = list(param_grid)

                    shuffler=check_random_state(int(random_state*(run+1)*(iFoldVal+1)))
                    shuffler.shuffle(param_grid)
                    results_files = {join(paths['cache'], f"{p['method']}_landmarks_based_learner_{p['nb_landmarks']}_dataset_{dataname}.pkl"): p \
                                                                                                            for p in param_grid}

                    results_to_compute = [dict({"output_file":f}, **p) for f, p in results_files.items() if not(exists(f))]

                    if results_to_compute:

                        parallel_func = partial(compute_landmarks_selection,
                                    dataset=dataset,
                                    C_range=hps['C'],
                                    gamma=gamma,
                                    random_state=int(random_state*(run+1)*(iFoldVal+1)),
                                    k=iFoldVal+1,
                                    nrun=run+1,
                                    deg=deg)

                        computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))

        # Learning
                    param_grid = ParameterGrid([{'algo': ['pb_da'], 'D': hps['landmarks_D'], 'method': args.landmarks_method, \
                                                                                       'nb_landmarks': hps['nb_landmarks']},
                                                   {'algo': ['pb'], 'D': hps['landmarks_D'], 'method': args.landmarks_method, \
                                                                                       'nb_landmarks': hps['nb_landmarks']},
                                                   {'algo': ['rbf'], 'method': args.landmarks_method, 'nb_landmarks': hps['nb_landmarks']}])
                    param_grid = list(param_grid)
                        
                    shuffler.shuffle(param_grid)

                    results_files = {join(paths[f"landmarks_based_{p['method']}"], f"{p['algo']}_{p['nb_landmarks']}_dataset_{dataname}" \
                                                        + (f"_{p['D']}.pkl" if 'D' in p else ".pkl")): p for p in param_grid}

                    results_to_compute = [dict({"output_file":f, "input_file": join(paths['cache'], \
                                                   f"{p['method']}_landmarks_based_learner_{p['nb_landmarks']}_dataset_{dataname}.pkl")}, **p) \
                                                                                    for f, p in results_files.items() if not(exists(f))]
                    if results_to_compute:
                        parallel_func = partial(compute_landmarks_based_DA,
                                beta_range=hps['beta'], beta_DA_range=hps['beta_da'], c_range=hps['c'], b_range=hps['b'], reverse=reversevalidation)

                        computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))
                        
if __name__ == '__main__':
    main()
