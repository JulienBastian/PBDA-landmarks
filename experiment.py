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

##Python file to run the toy experiments

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

    ##Source dataset 

    X_S_test, y_S_test, nul, nul= make_moons_da(6000, rotation=00, noise=0.1, random_state=random_state)
    
    degrees=[10,20,30,40,50,70,90]
    n_runs=10
    nbFoldValid=5
    for deg in degrees:
        nul, nul, X_T_test, y_T_test= make_moons_da(6000, rotation=deg, noise=0.1, random_state=random_state)
        for run in range(n_runs):
            X_Source_train, y_Source_train, X_Target_train, y_Target_train =make_moons_da(2000, rotation=deg, noise=0.1, random_state=int(random_state*(run+1)))
            skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True, random_state=int(random_state*(run+1))) 
            foldsTrainValidSource = list(skf.split(X_Source_train, y_Source_train)) 
            foldsTrainValidTarget = list(skf.split(X_Target_train, y_Target_train))
            for iFoldVal in range(nbFoldValid):
                    start_time=time.time()
                    idxTrainSource, idxValidSource = foldsTrainValidSource[iFoldVal]
                    idxTrainTarget, idxValidTarget = foldsTrainValidTarget[iFoldVal]
                    X_S_train=X_Source_train[idxTrainSource]
                    y_S_train=y_Source_train[idxTrainSource]

                    X_S_valid=X_Source_train[idxValidSource]
                    y_S_valid=y_Source_train[idxValidSource]

                    X_T_train=X_Target_train[idxTrainTarget]
                    y_T_train=y_Target_train[idxTrainTarget]

                    X_T_valid=X_Target_train[idxValidTarget]
                    y_T_valid=y_Target_train[idxValidTarget]


                    dataset = {'name': args.dataset,
                               'X_S_train': X_S_train, 'X_T_train': X_T_train, 'X_S_valid': X_S_valid, 'X_T_test': X_T_test,'X_S_test': X_S_test, 'X_T_valid': X_T_valid,
                               'y_S_train': y_S_train, 'y_T_train': y_T_train, 'y_S_valid': y_S_valid, 'y_T_test': y_T_test, 'y_S_test': y_S_test,'y_T_valid': y_T_valid}
                    hps = {'gamma': np.logspace(-7, 2, 10),
                            'C': [0.1, 10, 100],
                            'beta': np.logspace(-3, 3, 3),
                            'beta_da': [10.0, 1.0, 0.01],
                            'c':[10.0, 1.0, 0.01],
                            'b':[10.0, 1.0, 0.01],
                            'C_dalc' : [10.0, 1.0, 0.01],
                            'B_dalc' : [10.0, 1.0, 0.01],
                            'C_pbda' : [10.0, 1.0, 0.01],
                            'A_pbda' : [10.0, 1.0, 0.01],
                            'landmarks_percentage': [0.01, 0.05, 0.1, 0.15, 0.20, 0.25],
                            'nb_landmarks':[2,4,8,16],
                            'landmarks_D': [4, 8, 16],
                            'rho': [1.0, 0.1, 0.01, 0.001, 0.0001],
                            'greedy_kernel_N': 20000,
                            'greedy_kernel_D': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275,\
                               300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500, 1750, 2000, 2500, 3000,\
                               3500, 4000, 4500, 5000]}
                    
                    svm_file = join(paths['baseline'], f"svm_degree_{deg}_k_{iFoldVal+1}_run_{run+1}.pkl")
                    print(f"fold {iFoldVal+1}, run {run+1}, degree {deg}")
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

                    gamma = svm_results[0]["gamma"]
                    

                    tsvm_file = join(paths['baseline'], f"tsvm_degree_{deg}_k_{iFoldVal+1}_run_{run+1}.pkl")

                    learn_TSVM(dataset=dataset,
                                C_range=hps['C'],
                                gamma=gamma,
                                output_file=tsvm_file,
                                n_cpu=n_cpu,
                                k=iFoldVal+1,
                                nrun=run+1,
                                deg=deg,
                                random_state=int(random_state*(run+1)*(iFoldVal+1)),
                                reverse=reversevalidation)
                    
                    dalc_file = join(paths['baseline'], f"dalc_degree_{deg}_k_{iFoldVal+1}_run_{run+1}.pkl")

                    learn_dalc(dataset=dataset,
                                C_dalc_range=hps['C_dalc'],
                                B_dalc_range=hps['B_dalc'],
                                output_file=dalc_file,
                                k=iFoldVal+1,
                                nrun=run+1,
                                deg=deg,
                                random_state=int(random_state*(run+1)*(iFoldVal+1)),
                                reverse=reversevalidation)
                    
                    pbda_file = join(paths['baseline'], f"pbda_degree_{deg}_k_{iFoldVal+1}_run_{run+1}.pkl")

                    learn_pbda(dataset=dataset,
                                C_pbda_range=hps['C_pbda'],
                                A_pbda_range=hps['A_pbda'],
                                output_file=pbda_file,
                                k=iFoldVal+1,
                                nrun=run+1,
                                deg=deg,
                                random_state=int(random_state*(run+1)*(iFoldVal+1)),
                                reverse=reversevalidation)

                    


                    if "landmarks_based" in args.experiments:

                    # Initializing landmarks-based learners by selecting landmarks according to methods
                        param_grid = ParameterGrid([{'method': args.landmarks_method, 'nb_landmarks': hps['nb_landmarks']}])
                        param_grid = list(param_grid)

                        shuffler=check_random_state(int(random_state*(run+1)*(iFoldVal+1)))
                        shuffler.shuffle(param_grid)
                        results_files = {join(paths['cache'], f"{p['method']}_landmarks_based_learner_{p['nb_landmarks']}_degree_{deg}_k_{iFoldVal+1}_run_{run+1}.pkl"): p \
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

                        results_files = {join(paths[f"landmarks_based_{p['method']}"], f"{p['algo']}_{p['nb_landmarks']}_degree_{deg}_k_{iFoldVal+1}_run_{run+1}" \
                                                        + (f"_{p['D']}.pkl" if 'D' in p else ".pkl")): p for p in param_grid}

                        results_to_compute = [dict({"output_file":f, "input_file": join(paths['cache'], \
                                                   f"{p['method']}_landmarks_based_learner_{p['nb_landmarks']}_degree_{deg}_k_{iFoldVal+1}_run_{run+1}.pkl")}, **p) \
                                                                                    for f, p in results_files.items() if not(exists(f))]
                        if results_to_compute:
                            parallel_func = partial(compute_landmarks_based_DA,
                                    beta_range=hps['beta'], beta_DA_range=hps['beta_da'], c_range=hps['c'], b_range=hps['b'], reverse=reversevalidation)

                            computed_results = list(Pool(processes=n_cpu).imap(parallel_func, results_to_compute))
                        end_time=time.time()

                        elapsed_time=end_time-start_time
                        print(f"temps pour la fold : ", elapsed_time)

    print("### DONE ###")

if __name__ == '__main__':
    main()
