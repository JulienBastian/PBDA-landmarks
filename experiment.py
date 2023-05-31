import argparse
from functools import partial
import multiprocessing
from multiprocessing import Pool

import os
from os.path import join, abspath, dirname, exists
from os import makedirs

import pickle
import numpy as np

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split, ParameterGrid

from sklearn.datasets import make_moons

from pbrff.data_loader import DataLoader
from pbrff.baseline import learn_svm
from pbrff.greedy_kernel import GreedyKernelLearner, compute_greedy_kernel
from pbrff.landmarks_based import compute_landmarks_selection, compute_landmarks_based, compute_landmarks_based_DA

import warnings
warnings.filterwarnings("ignore")

##### Datasets DA

from dataset import *

RESULTS_PATH = os.environ.get('PBRFF_RESULTS_DIR', join(dirname(abspath(__file__)), "results"))

def make_moons_da(n_samples=50, rotation=50, noise=0.15, random_state=0):
    Xs, ys = make_moons(n_samples=n_samples,
                        noise=noise,
                        random_state=random_state)
    Xs[:, 0] -= 0.5
    theta = np.radians(-rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array(
        ((cos_theta, -sin_theta),
         (sin_theta, cos_theta))
    )
    Xt = Xs.dot(rot_matrix)
    yt = ys
    return Xs, ys, Xt, yt

def main():
    parser = argparse.ArgumentParser(description="PAC-Bayes RFF Experiment")
    parser.add_argument('-d', '--dataset', type=str, default="breast")
    ###### datasets DA
    parser.add_argument('-dsource', '--datasetsource', type=str, default="breast")
    parser.add_argument('-dtarget', '--datasettarget', type=str, default="breast")
    ######
    parser.add_argument('-e', '--experiments', type=str, nargs='+', default=["landmarks_based"])
    parser.add_argument('-l', '--landmarks-method', type=str, nargs='+', default=["random"])
    parser.add_argument('-n', '--n-cpu', type=int, default=-1)
    args = parser.parse_args()

    # Setting random seed for repeatability
    random_seed = 42
    random_state = check_random_state(random_seed)

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

    #two moons dataset

    ###### datasets DA

    """source_data = dataset_from_svmlight_file(args.datasetsource)
    target_data = dataset_from_svmlight_file(args.datasettarget, source_data.get_nb_features())
    source_data.reshape_features(target_data.get_nb_features())

    X_source=source_data.X
    y_source=source_data.Y

    X_target=target_data.X
    y_target=target_data.Y"""
    ######


    # Loading dataset
    #dataloader = DataLoader(random_state=random_state)
    #X_source, X_target, y_source, y_target = dataloader.load(args.dataset)

    X_source, y_source, X_target, y_target= make_moons_da(300, rotation=50, noise=0.05, random_state=random_state)
    
    #X_source, X_trash, y_source, y_trash = train_test_split(X_source, y_source, test_size=0.8, random_state=random_state)
    #X_target, X_trash, y_target, y_trash = train_test_split(X_target, y_target, test_size=0.8, random_state=random_state)


    X_S_train, X_S_valid, y_S_train, y_S_valid = train_test_split(X_source, y_source, test_size=0.2, random_state=random_state)
    X_T_train, X_T_test, y_T_train, y_T_test = train_test_split(X_target, y_target, test_size=0.2, random_state=random_state)
    X_T_train, X_T_valid, y_T_train, y_T_valid = train_test_split(X_T_train, y_T_train, test_size=0.2, random_state=random_state)
    
    #dataset = {'name': args.dataset,
    #           'X_train': X_S_train, 'X_valid': X_S_valid, 'X_test': X_T_test,
    #           'y_train': y_S_train, 'y_valid': y_S_valid, 'y_test': y_T_test}
    dataset = {'name': args.dataset,
               'X_S_train': X_S_train, 'X_T_train': X_T_train, 'X_S_valid': X_S_valid, 'X_T_test': X_T_test, 'X_T_valid': X_T_valid,
               'y_S_train': y_S_train, 'y_T_train': y_T_train, 'y_S_valid': y_S_valid, 'y_T_test': y_T_test,'y_T_valid': y_T_valid}


    # HPs for landmarks-based and greedy kernel learning experiments
    hps = {'gamma': np.logspace(-7, 2, 10),
           'C': np.logspace(-2, 4, 5),
           'beta': np.logspace(-3, 3, 5),
           'beta_da': np.logspace(-3, 3, 5),
           'c':[10.0, 1.0, 0.1, 0.01],
           'b':[10.0, 1.0, 0.1, 0.01],
           'landmarks_percentage': [0.01, 0.05, 0.1, 0.15, 0.20, 0.25],
           'nb_landmarks':[2,4,6,12,16,20],
           'landmarks_D': [8, 16, 32, 64],
           'rho': [1.0, 0.1, 0.01, 0.001, 0.0001],
           'greedy_kernel_N': 20000,
           'greedy_kernel_D': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275,\
                               300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500, 1750, 2000, 2500, 3000,\
                               3500, 4000, 4500, 5000]}

    ### Experiments ###

    # Baseline (SVM)
    svm_file = join(paths['baseline'], "svm.pkl")
    if not(exists(svm_file)):
        learn_svm(dataset=dataset,
                  C_range=hps['C'],
                  gamma_range=hps['gamma'],
                  output_file=svm_file,
                  n_cpu=n_cpu,
                  random_state=random_state)

    with open(svm_file, 'rb') as in_file:
        svm_results = pickle.load(in_file)

    gamma = svm_results[0]["gamma"]


    # Landmarks-based learning
    if "landmarks_based" in args.experiments:

        # Initializing landmarks-based learners by selecting landmarks according to methods
        param_grid = ParameterGrid([{'method': args.landmarks_method, 'nb_landmarks': hps['nb_landmarks']}])
        param_grid = list(param_grid)

        random_state.shuffle(param_grid)
        results_files = {join(paths['cache'], f"{p['method']}_landmarks_based_learner_{p['nb_landmarks']}.pkl"): p \
                                                                                                            for p in param_grid}
        results_to_compute = [dict({"output_file":f}, **p) for f, p in results_files.items() if not(exists(f))]

        if results_to_compute:

            parallel_func = partial(compute_landmarks_selection,
                                    dataset=dataset,
                                    C_range=hps['C'],
                                    gamma=gamma,
                                    random_state=random_state)

            computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))

        # Learning
        param_grid = ParameterGrid([{'algo': ['pb_da'], 'D': hps['landmarks_D'], 'method': args.landmarks_method, \
                                                                                       'nb_landmarks': hps['nb_landmarks']},
                                    {'algo': ['pb'], 'D': hps['landmarks_D'], 'method': args.landmarks_method, \
                                                                                       'nb_landmarks': hps['nb_landmarks']},
                                    {'algo': ['rbf'], 'method': args.landmarks_method, 'nb_landmarks': hps['nb_landmarks']}])
        param_grid = list(param_grid)
        random_state.shuffle(param_grid)
        results_files = {join(paths[f"landmarks_based_{p['method']}"], f"{p['algo']}_{p['nb_landmarks']}" \
                                                        + (f"_{p['D']}.pkl" if 'D' in p else ".pkl")): p for p in param_grid}

        results_to_compute = [dict({"output_file":f, "input_file": join(paths['cache'], \
                                    f"{p['method']}_landmarks_based_learner_{p['nb_landmarks']}.pkl")}, **p) \
                                                                                    for f, p in results_files.items() if not(exists(f))]
        if results_to_compute:
            parallel_func = partial(compute_landmarks_based_DA,
                                    beta_range=hps['beta'], beta_DA_range=hps['beta_da'], c_range=hps['c'], b_range=hps['b'], C_range=hps['C'])

            computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))

    # Greedy Kernel Learning
    if "greedy_kernel" in args.experiments:

        # Initializing greedy kernel learner
        greedy_kernel_learner_cache_file = join(paths['cache'], "greedy_kernel_learner.pkl")
        if not exists(greedy_kernel_learner_cache_file):
            greedy_kernel_learner = GreedyKernelLearner(dataset, hps['C'], gamma, hps['greedy_kernel_N'], random_state)
            greedy_kernel_learner.sample_omega()
            greedy_kernel_learner.compute_loss()

            with open(greedy_kernel_learner_cache_file, 'wb') as out_file:
                pickle.dump(greedy_kernel_learner, out_file, protocol=4)

        param_grid = ParameterGrid([{'algo': ["pbrff"], 'param': hps['beta']},
                                    {'algo': ["okrff"], 'param': hps['rho']},
                                    {'algo': ["rff"]}])

        param_grid = list(param_grid)
        random_state.shuffle(param_grid)
        results_files = {join(paths['greedy_kernel'], f"{p['algo']}" + (f"_{p['param']}.pkl" if 'param' in p else ".pkl")): p \
                                                                                                            for p in param_grid}
        results_to_compute = [dict({"output_file":f}, **p) for f, p in results_files.items() if not(exists(f))]

        if results_to_compute:
            parallel_func = partial(compute_greedy_kernel,
                                    greedy_kernel_learner_file=greedy_kernel_learner_cache_file,
                                    gamma=gamma,
                                    D_range=hps['greedy_kernel_D'],
                                    random_state=random_state)

            computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))
            
    print("### DONE ###")

if __name__ == '__main__':
    main()
