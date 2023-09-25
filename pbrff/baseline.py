import pickle
import time
import numpy as np

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from .TSVM.tsvm import SKTSVM
from .kernel import Kernel
from .dalc import Dalc
from .pbda import Pbda

def learn_svm(dataset, C_range, gamma_range, output_file, n_cpu, k, nrun, deg, random_state):
    """Learn a SVM baseline.

    Using a validation set, hyperparameters C and gamma are selected
    from given ranges through grid search.

    Parameters
    ----------
    dataset: dict
        The dataset as a dictionnary with the following keys:
        X_S_train, X_T_valid, X_T_test, y_S_train, y_T_valid, y_T_test, name.

    C_range: list
        C values range to search from (SVM's penalty parameter).

    gamma_range: list
        Gamma values range to search from (RBF kernel's bandwidth parameter).

    output_file: str
        File path to save results with pickle

    n_cpu: int
        The number of CPUs to use during the grid search.
    
    k : int
        Current fold.
    
    deg : 
        Degree of rotation if toy experiment on two intertwined moons dataset.

    random_state: instance of RandomState
        Random state for all random operations.

    """
    print("Computing SVM baseline")
    start_time = time.time()

    # Defining the validation set for GridSearchCV
    X = np.concatenate((dataset['X_S_train'], dataset['X_S_valid']))
    y = np.concatenate((dataset['y_S_train'], dataset['y_S_valid']))
    valid_fold = np.zeros(len(y))
    valid_fold[:len(dataset['y_S_train'])] = -1
    valid_split = PredefinedSplit(test_fold=valid_fold)

    # Grid search using a validation set
    param_grid = [{'C': C_range, 'gamma': gamma_range}]
    gs = GridSearchCV(SVC(kernel='rbf', random_state=random_state),
                      param_grid=param_grid,
                      n_jobs=n_cpu,
                      cv=valid_split,
                      refit=False)
    gs.fit(X, y)

    # SVM classifier training using best hps values selected
    clf = SVC(kernel='rbf', gamma=gs.best_params_['gamma'], C=gs.best_params_['C'], random_state=random_state)
    clf.fit(dataset['X_S_train'], dataset['y_S_train'])
    train_time = (time.time() - start_time)

    # Computing relevant metrics

    val_s_err = 1 - accuracy_score(dataset['y_S_valid'], clf.predict(dataset['X_S_valid']))
    val_t_err = 1 - accuracy_score(dataset['y_T_valid'], clf.predict(dataset['X_T_valid']))
    train_err = 1 - accuracy_score(dataset['y_S_train'], clf.predict(dataset['X_S_train']))
    test_err_t = 1 - accuracy_score(dataset['y_T_test'], clf.predict(dataset['X_T_test']))
    f1_t = f1_score(dataset['y_T_test'], clf.predict(dataset['X_T_test']))
    test_err_s = 1 - accuracy_score(dataset['y_S_test'], clf.predict(dataset['X_S_test']))
    f1_s = f1_score(dataset['y_S_test'], clf.predict(dataset['X_S_test']))

    # Logging metrics and informations
    results = [dict([("dataset", dataset['name']), ("exp", 'baseline'), ("algo", 'SVM'),\
                    ("C", gs.best_params_['C']), ("gamma", gs.best_params_['gamma']), ("time", train_time),\
                    ("train_error", train_err),("val_s_error", val_s_err), ("val_t_error", val_t_err), ("test_error_t", test_err_t), \
                    ("f1_t", f1_t), ("test_error_s", test_err_s), ("f1_s", f1_s), ("k", k), ("run", nrun), ("degree", deg)])]

    with open(output_file, 'wb') as out_file:
        pickle.dump(results, out_file, protocol=4)


def learn_TSVM(dataset, C_range, gamma, output_file, n_cpu, k, nrun, deg, random_state, reverse):
    """Learn a TSVM baseline.

    Using a validation set, hyperparameters C and gamma are selected
    from given ranges through grid search.

    Parameters
    ----------
    dataset: dict
        The dataset as a dictionnary with the following keys:
        X_S_train, X_T_valid, X_T_test, y_S_train, y_T_valid, y_T_test, name.

    C_range: list
        C values range to search from (SVM's penalty parameter).

    gamma_range: list
        Gamma values range to search from (RBF kernel's bandwidth parameter).

    output_file: str
        File path to save results with pickle
      
    k : int
        Current fold.
    
    deg : 
        Degree of rotation if toy experiment on two intertwined moons dataset.

    random_state: instance of RandomState
        Random state for all random operations.
    
    reverse:
        "yes" if reverse validation should be performed, "no" otherwise.

    """
    print("Computing TSVM")

    # Defining the validation set for GridSearchCV
    start_time = time.time()

    y_S_train=np.array(dataset['y_S_train'])
    y_T_train=np.array(dataset['y_T_train'])

    y_S_test=np.array(dataset['y_S_test'])
    y_T_test=np.array(dataset['y_T_test'])

    y_T_train[:]=-1
    y_train=np.concatenate((y_S_train, y_T_train))
    X_train=np.concatenate((dataset['X_S_train'], dataset['X_T_train']))
    
    if reverse=="no" or reverse == "yes":
        X = np.concatenate((dataset['X_S_train'], dataset['X_T_train'], dataset['X_S_valid']))
        y = np.concatenate((dataset['y_S_train'], y_T_train, dataset['y_S_valid']))
        valid_fold = np.zeros(len(y))
        valid_fold[:len(dataset['y_S_train'])+len(y_T_train)] = -1
        valid_split = PredefinedSplit(test_fold=valid_fold)
        
        # Grid search using a validation set
        param_grid = [{'C': C_range, 'gamma': [gamma]}]
        gs = GridSearchCV(SKTSVM(kernel='rbf', random_state=random_state),
                        param_grid=param_grid,
                        n_jobs=n_cpu,
                        cv=valid_split,
                        refit=False)
        gs.fit(X, y)

        # SVM classifier training using best hps values selected
        clf = SKTSVM(kernel='rbf', gamma=gamma, C=gs.best_params_['C'], random_state=random_state)
        clf.fit(X_train, y_train)
        train_time = (time.time() - start_time) 

        # Computing relevant metrics
        val_s_err = 1 - accuracy_score(dataset['y_S_valid'], clf.predict(dataset['X_S_valid']))
        val_t_err = 1 - accuracy_score(dataset['y_T_valid'], clf.predict(dataset['X_T_valid']))
        train_err = 1 - accuracy_score(dataset['y_S_train'], clf.predict(dataset['X_S_train']))
        test_err_t = 1 - accuracy_score(dataset['y_T_test'], clf.predict(dataset['X_T_test']))
        f1_t = f1_score(dataset['y_T_test'], clf.predict(dataset['X_T_test']))
        test_err_s = 1 - accuracy_score(dataset['y_S_test'], clf.predict(dataset['X_S_test']))
        f1_s = f1_score(dataset['y_S_test'], clf.predict(dataset['X_S_test']))


    # Logging metrics and informations
    results = [dict([("dataset", dataset['name']), ("exp", 'baseline'), ("algo", 'TSVM'),\
                    ("C", gs.best_params_['C']), ("gamma", gs.best_params_['gamma']), ("time", train_time),\
                    ("train_error", train_err),("val_s_error", val_s_err), ("val_t_error", val_t_err), ("test_error_t", test_err_t), \
                    ("f1_t", f1_t), ("test_error_s", test_err_s), ("f1_s", f1_s), ("k", k), ("run", nrun), ("degree", deg)])]

    with open(output_file, 'wb') as out_file:
        pickle.dump(results, out_file, protocol=4)



def learn_dalc(dataset, C_dalc_range, B_dalc_range, output_file, k, nrun, deg, random_state, reverse):
    """Learn a DALC classifier.

    Using a validation set, hyperparameters C and gamma are selected
    from given ranges through grid search.

    Parameters
    ----------
    dataset: dict
        The dataset as a dictionnary with the following keys:
        X_S_train, X_T_valid, X_T_test, y_S_train, y_T_valid, y_T_test, name.

    C_dalc_range: list
        C_dalc values range to search from (disagreement tradeoff parameters).

    B_dalc_range: list
        B_dalc values range to search from (joint error tradeoff parameters).

    output_file: str
        File path to save results with pickle
  
    k : int
        Current fold.
    
    deg : 
        Degree of rotation if toy experiment on two intertwined moons dataset.

    random_state: instance of RandomState
        Random state for all random operations.
    
    reverse:
        "yes" if reverse validation should be performed, "no" otherwise.

    """
    print("computing DALC")
    start_time = time.time()

    y_S_train=np.array(dataset['y_S_train'])
    y_T_train=np.array(dataset['y_T_train'])

    y_S_train[y_S_train==0]=-1
    y_T_train[y_T_train==0]=-1

    y_S_test=np.array(dataset['y_S_test'])
    y_T_test=np.array(dataset['y_T_test'])

    y_S_test[y_S_test==0]=-1
    y_T_test[y_T_test==0]=-1

    y_S_valid=np.array(dataset['y_S_valid'])
    y_T_valid=np.array(dataset['y_T_valid'])

    y_S_valid[y_S_valid==0]=-1
    y_T_valid[y_T_valid==0]=-1

    dataset['X_S_train']
    dataset['X_T_train']

    dataset['X_S_test']
    dataset['X_T_test']

    dataset['X_S_valid']
    dataset['X_T_valid']

    kernel = Kernel('linear')
    val_s_err=None
    val_t_err=None
    reverse_val=None

    C_search = []
    if reverse=="no":
        for C in C_dalc_range:
            for B in B_dalc_range:

                clf = Dalc(C=C, B=B, verbose=False)
                classifier = clf.learn(dataset['X_S_train'], dataset['X_T_train'], y_S_train, y_T_train, kernel=kernel)
                y_s_valid_pred = classifier.predict(dataset['X_S_valid'])
                y_s_valid_pred[y_s_valid_pred>0]=1
                y_s_valid_pred[y_s_valid_pred<=0]=-1
                val_s_err = 1 - accuracy_score(y_S_valid,y_s_valid_pred )

                y_t_valid_pred = classifier.predict(dataset['X_T_valid'])
                y_t_valid_pred[y_t_valid_pred>0]=1
                y_t_valid_pred[y_t_valid_pred<=0]=-1

                val_t_err = 1 - accuracy_score(y_T_valid, y_t_valid_pred)

                train_pred= classifier.predict(dataset['X_S_train'])
                train_pred[train_pred>0]=1
                train_pred[train_pred<=0]=-1
                train_err = 1 - accuracy_score(y_S_train, train_pred)

                y_t_pred = classifier.predict(dataset['X_T_test'])
                y_t_pred[y_t_pred>0]=1
                y_t_pred[y_t_pred<=0]=-1

                test_err_t = 1 - accuracy_score(y_T_test, y_t_pred)
                f1_t = f1_score(y_T_test, y_t_pred)

                y_s_pred = classifier.predict(dataset['X_S_test'])
                y_s_pred[y_s_pred>0]=1
                y_s_pred[y_s_pred<=0]=-1

                test_err_s = 1 - accuracy_score(dataset['y_S_test'], y_s_pred)
                f1_s = f1_score(y_S_test, y_s_pred)
                C_search.append((val_s_err, val_t_err, train_err, test_err_t, f1_t, y_s_pred, test_err_s, f1_s, C, B))

        val_s_err, val_t_err, train_err, test_err_t, f1_t, y_s_pred, test_err_s, f1_s, C, B = sorted(C_search, key=lambda x: x[0])[0]
        
        end_time=time.time()             

    
    elif reverse=="yes":
        for C in C_dalc_range:
            for B in B_dalc_range:
                clf = Dalc(C=C, B=B, verbose=False)
                classifier = clf.learn(X_source=dataset['X_S_train'], X_target=dataset['X_T_train'], y_source=y_S_train, y_target=y_T_train, kernel=kernel)
                ##ici reverse val
                autolabeled = classifier.predict(dataset['X_T_train'])
                autolabeled[autolabeled>0]=1
                autolabeled[autolabeled<=0]=-1

                reverse_clf = Dalc(C=C, B=B, verbose=False)
                reverse_classifier = reverse_clf.learn(X_source=dataset['X_T_train'], X_target=dataset['X_S_train'], y_source=autolabeled, y_target=y_S_train, kernel=kernel)
                
                reverse_val_pred=reverse_classifier.predict(dataset['X_S_valid'])
                reverse_val_pred[reverse_val_pred>0]=1
                reverse_val_pred[reverse_val_pred<=0]=-1
                reverse_val=1 - accuracy_score(y_S_valid, reverse_val_pred)

                train_pred= classifier.predict(dataset['X_S_train'])
                train_pred[train_pred>0]=1
                train_pred[train_pred<=0]=-1

                train_err = 1 - accuracy_score(y_S_train,train_pred)

                y_t_pred = classifier.predict(dataset['X_T_test'])
                y_t_pred[y_t_pred>0]=1
                y_t_pred[y_t_pred<=0]=-1

                test_err_t = 1 - accuracy_score(y_T_test, y_t_pred)
                f1_t = f1_score(y_T_test, y_t_pred)

                y_s_pred = classifier.predict(dataset['X_S_test'])
                y_s_pred[y_s_pred>0]=1
                y_s_pred[y_s_pred<=0]=-1
                test_err_s = 1 - accuracy_score(dataset['y_S_test'], y_s_pred)
                f1_s = f1_score(y_S_test, y_s_pred)
                C_search.append((reverse_val, train_err, test_err_t, f1_t, y_s_pred, test_err_s, f1_s, C, B))

            reverse_val, train_err, test_err_t, f1_t, y_s_pred, test_err_s, f1_s, C, B = sorted(C_search, key=lambda x: x[0])[0]
        end_time=time.time()   

    train_time = (time.time() - start_time)

    results = [dict([("dataset", dataset['name']), ("exp", 'baseline'), ("algo", 'DALC'),\
                    ("C_dalc", C), ("time", train_time),\
                    ("train_error", train_err),("val_s_error", val_s_err), ("val_t_error", val_t_err), ("reverse_val_error", reverse_val), ("test_error_t", test_err_t), \
                    ("f1_t", f1_t), ("test_error_s", test_err_s), ("f1_s", f1_s), ("k", k), ("run", nrun), ("degree", deg)])]
    
    with open(output_file, 'wb') as out_file:
        pickle.dump(results, out_file, protocol=4)


def learn_pbda(dataset, C_pbda_range, A_pbda_range, output_file, k, nrun, deg, random_state, reverse):
    """Learn a PBDA classifier.

    Using a validation set, hyperparameters C and gamma are selected
    from given ranges through grid search.

    Parameters
    ----------
    dataset: dict
        The dataset as a dictionnary with the following keys:
        X_S_train, X_T_valid, X_T_test, y_S_train, y_T_valid, y_T_test, name.

    C_pbda_range: list
        C_pbda values range to search from (disagreement tradeoff parameters).

    B_pbda_range: list
        B_pbda values range to search from (joint error tradeoff parameters).

    output_file: str
        File path to save results with pickle
  
    k : int
        Current fold.
    
    deg : 
        Degree of rotation if toy experiment on two intertwined moons dataset.

    random_state: instance of RandomState
        Random state for all random operations.
    
    reverse:
        "yes" if reverse validation should be performed, "no" otherwise.

    """
    print("computing PBDA")
    start_time = time.time()

    y_S_train=np.array(dataset['y_S_train'])
    y_T_train=np.array(dataset['y_T_train'])

    y_S_train[y_S_train==0]=-1
    y_T_train[y_T_train==0]=-1

    y_S_test=np.array(dataset['y_S_test'])
    y_T_test=np.array(dataset['y_T_test'])

    y_S_test[y_S_test==0]=-1
    y_T_test[y_T_test==0]=-1

    y_S_valid=np.array(dataset['y_S_valid'])
    y_T_valid=np.array(dataset['y_T_valid'])

    y_S_valid[y_S_valid==0]=-1
    y_T_valid[y_T_valid==0]=-1

    dataset['X_S_train']
    dataset['X_T_train']

    dataset['X_S_test']
    dataset['X_T_test']

    dataset['X_S_valid']
    dataset['X_T_valid']

    kernel = Kernel('linear')
    val_s_err=None
    val_t_err=None
    reverse_val=None

    C_search = []
    if reverse=="no":
        for C in C_pbda_range:
            for A in A_pbda_range:

                clf = Pbda(C=C, A=A, verbose=False)
                classifier = clf.learn(dataset['X_S_train'], dataset['X_T_train'], y_S_train, y_T_train, kernel=kernel)
                y_s_valid_pred = classifier.predict(dataset['X_S_valid'])
                y_s_valid_pred[y_s_valid_pred>0]=1
                y_s_valid_pred[y_s_valid_pred<=0]=-1
                val_s_err = 1 - accuracy_score(y_S_valid,y_s_valid_pred )

                y_t_valid_pred = classifier.predict(dataset['X_T_valid'])
                y_t_valid_pred[y_t_valid_pred>0]=1
                y_t_valid_pred[y_t_valid_pred<=0]=-1

                val_t_err = 1 - accuracy_score(y_T_valid, y_t_valid_pred)

                train_pred= classifier.predict(dataset['X_S_train'])
                train_pred[train_pred>0]=1
                train_pred[train_pred<=0]=-1
                train_err = 1 - accuracy_score(y_S_train, train_pred)

                y_t_pred = classifier.predict(dataset['X_T_test'])
                y_t_pred[y_t_pred>0]=1
                y_t_pred[y_t_pred<=0]=-1

                test_err_t = 1 - accuracy_score(y_T_test, y_t_pred)
                f1_t = f1_score(y_T_test, y_t_pred)

                y_s_pred = classifier.predict(dataset['X_S_test'])
                y_s_pred[y_s_pred>0]=1
                y_s_pred[y_s_pred<=0]=-1

                test_err_s = 1 - accuracy_score(dataset['y_S_test'], y_s_pred)
                f1_s = f1_score(y_S_test, y_s_pred)
                C_search.append((val_s_err, val_t_err, train_err, test_err_t, f1_t, y_s_pred, test_err_s, f1_s, C, A))

        val_s_err, val_t_err, train_err, test_err_t, f1_t, y_s_pred, test_err_s, f1_s, C, A = sorted(C_search, key=lambda x: x[0])[0]
        
        end_time=time.time()             

    
    elif reverse=="yes":
        for C in C_pbda_range:
            for A in A_pbda_range:
                clf = Pbda(C=C, A=A, verbose=False)
                classifier = clf.learn(X_source=dataset['X_S_train'], X_target=dataset['X_T_train'], y_source=y_S_train, y_target=y_T_train, kernel=kernel)
                ##ici reverse val
                autolabeled = classifier.predict(dataset['X_T_train'])
                autolabeled[autolabeled>0]=1
                autolabeled[autolabeled<=0]=-1

                reverse_clf = Pbda(C=C, A=A, verbose=False)
                reverse_classifier = reverse_clf.learn(X_source=dataset['X_T_train'], X_target=dataset['X_S_train'], y_source=autolabeled, y_target=y_S_train, kernel=kernel)
                
                reverse_val_pred=reverse_classifier.predict(dataset['X_S_valid'])
                reverse_val_pred[reverse_val_pred>0]=1
                reverse_val_pred[reverse_val_pred<=0]=-1
                reverse_val=1 - accuracy_score(y_S_valid, reverse_val_pred)

                train_pred= classifier.predict(dataset['X_S_train'])
                train_pred[train_pred>0]=1
                train_pred[train_pred<=0]=-1

                train_err = 1 - accuracy_score(y_S_train,train_pred)

                y_t_pred = classifier.predict(dataset['X_T_test'])
                y_t_pred[y_t_pred>0]=1
                y_t_pred[y_t_pred<=0]=-1

                test_err_t = 1 - accuracy_score(y_T_test, y_t_pred)
                f1_t = f1_score(y_T_test, y_t_pred)

                y_s_pred = classifier.predict(dataset['X_S_test'])
                y_s_pred[y_s_pred>0]=1
                y_s_pred[y_s_pred<=0]=-1
                test_err_s = 1 - accuracy_score(dataset['y_S_test'], y_s_pred)
                f1_s = f1_score(y_S_test, y_s_pred)
                C_search.append((reverse_val, train_err, test_err_t, f1_t, y_s_pred, test_err_s, f1_s, C, A))

            reverse_val, train_err, test_err_t, f1_t, y_s_pred, test_err_s, f1_s, C, A = sorted(C_search, key=lambda x: x[0])[0]
        end_time=time.time()   

    train_time = (time.time() - start_time) 

    results = [dict([("dataset", dataset['name']), ("exp", 'baseline'), ("algo", 'PBDA'),\
                    ("C_pbda", C), ("time", train_time),\
                    ("train_error", train_err),("val_s_error", val_s_err), ("val_t_error", val_t_err), ("reverse_val_error", reverse_val), ("test_error_t", test_err_t), \
                    ("f1_t", f1_t), ("test_error_s", test_err_s), ("f1_s", f1_s), ("k", k), ("run", nrun), ("degree", deg)])]
    
    with open(output_file, 'wb') as out_file:
        pickle.dump(results, out_file, protocol=4)
