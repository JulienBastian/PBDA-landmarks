from math import ceil, sqrt, exp, log

import time 

import pickle
import numpy as np

from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

from pbrff.landmarks_selector import LandmarksSelector

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



def reverse_validation(dataset, param, D, gamma, nb_landmarks, land_method, autolabeled, algo, random_state):

    X_S_train=dataset["X_T_train"]
    y_S_train=autolabeled

    X_T_train=dataset["X_S_train"]
    y_T_train=dataset["y_S_train"]

    X_T_valid=dataset["X_S_valid"]
    y_T_valid=dataset["y_S_valid"]

    reverse_data = {'X_S_train': X_S_train, 'X_T_train': X_T_train,'X_T_valid': X_T_valid,
            'y_S_train': y_S_train, 'y_T_train': y_T_train,'y_T_valid': y_T_valid}
                    

    landmarks_X, landmarks_y=select_landmarks(reverse_data["X_S_train"], reverse_data["y_S_train"], nb_landmarks, landmarks_selection_method=land_method, random_state=random_state)

    ns, d = reverse_data['X_S_train'].shape

    D=D

    Omega = random_state.randn(nb_landmarks, d, D) / (1. / sqrt(2 * gamma))

    if algo == "pb_da":
        score=pbda_landmark(landmarks_X, landmarks_y, reverse_data, param, D, gamma, nb_landmarks, landmarks_selection_method=land_method,Omega=Omega, random_state=random_state)

    if algo=="pb":
        score= pb_landmark(landmarks_X, landmarks_y, reverse_data, param, D, gamma, nb_landmarks, landmarks_selection_method=land_method,Omega=Omega, random_state=random_state)

    return score


##Whole learning procedure for PBDA-landmarks
def pbda_landmark(landmarks_X, landmarks_y, reverse_data, param, D, gamma, nb_landmarks,landmarks_selection_method, Omega, random_state):

    ns, d = reverse_data['X_S_train'].shape
    nt, d = reverse_data['X_T_train'].shape

    

    loss = []
    # Computing loss for each landmarks

    #Compute loss per landmark
    for i in range(nb_landmarks):
        transformed_X = transform_cos(Omega[i], reverse_data['X_S_train'] - landmarks_X[i])

        lambda_y = -np.ones(ns)
        lambda_y[(reverse_data['y_S_train'] == landmarks_y[i])] = 1

        landmark_loss = lambda_y @ transformed_X

        # For the clustering method, landmarks are not sampled directly from dataset
        if landmarks_selection_method == "clustering":
            landmark_loss = landmark_loss / (ns)

        # For the random method, case where X_i == landmark needs to be substracted
        elif landmarks_selection_method == "random":
            landmark_loss = (landmark_loss - 1) / (ns - 1)

        landmark_loss = (1 - landmark_loss) / 2
        loss.append(landmark_loss)

    loss_DA = np.array(loss)


    #compute disagreement
    disagreement=[]
        
        # Computing loss for each landmarks
             
    for i in range(nb_landmarks):
        list_L=[]
        #matrice contenant les hypothéses pour chaque point du target pour un landmark [i]
        transformed_X = transform_cos(Omega[i], reverse_data['X_T_train'] - landmarks_X[i])

        for j in range(nt):

            h=transformed_X[j,] #every hypothesis for a fixed point

            H=np.atleast_2d(h).T * h 

            H[np.triu_indices(D, k = 0)]=0 
            L=(1/2)*(1-H) 
            list_L.append(L)

        landmark_disagreement= 1/nt *(sum(list_L))

        disagreement.append(landmark_disagreement)

    disagreement = np.array(disagreement) 

    #compute joint error
    joint_error = []
        
        # Computing loss for each landmarks
    for i in range(nb_landmarks):
        list_L=[]
        transformed_X = transform_cos(Omega[i], reverse_data['X_S_train'] - landmarks_X[i])

        lambda_y = -np.ones(ns)
        lambda_y[(reverse_data['y_S_train'] == landmarks_y[i])] = 1
        to_stack=lambda_y

        for j in range(D-1):
            lambda_y = np.vstack((lambda_y,to_stack))

        unique_error=np.multiply(np.transpose(lambda_y), transformed_X)

        loss_by_hypothesis=(1/2)*(1-unique_error)

        for j in range(ns):
            l=loss_by_hypothesis[j,]
            L=np.atleast_2d(l).T*l
            L[np.triu_indices(D, k = 0)]=0
            list_L.append(L)
            
            
        if landmarks_selection_method == "clustering":
                
            landmark_joint_error=1/ns * (sum(list_L))
            landmark_joint_error = landmark_joint_error / (ns)

        # For the random method, case where X_i == landmark needs to be substracted
        elif landmarks_selection_method == "random":
            L=-np.ones((D, D))
            L[np.triu_indices(D, k = 0)]=0
            list_L.append(L)

            landmark_joint_error=1/(ns - 1) * (sum(list_L))
                

        joint_error.append(landmark_joint_error)
    joint_error = np.array(joint_error)
    
    #compute Q
    c=param["c"]
    b=param["b"]
    beta_da = param["beta_da"]
    c_plus=c/(1-exp(-c))
    b_plus=beta_da*b/(1-exp(-b))
    def kl_divergence(q):
        epsilon=0.0001
        q=abs(q) #even with constrained optimization q can be slightly negative
        return sum(q[i] * log((q[i]/(1/D)) + epsilon) for i in range(len(q)))

    
    

    Q_list=[]
    for i in range(nb_landmarks):
        Q=np.repeat(1/D, D)
        Q=np.array(Q)
        func= lambda x: c_plus * 1/2 * x @ disagreement[i] @ x + b_plus * x @ joint_error[i] @ x + ((c_plus/(ns*c))+(b_plus/(ns*b)))*2*kl_divergence(x)
        res=minimize(func, Q, bounds=Bounds(lb=0.01, ub=1, keep_feasible=False), constraints=LinearConstraint(np.ones(len(Q)), lb=1.0, ub=1.0, keep_feasible=False), method='trust-constr')
        Q=res.x
        # Computing Q
        Q_list.append(Q)
    Q_DA = Q_list

    ## PB_da mapping
    def pb_mapping_DA(X, nb_landmarks, Omega, landmarks_X, Q_DA):
        """PAC-Bayesian Domain Adaptation landmarks-based mapping of X according to computed pseudo-posterior Q distribution.

        Parameters
        ----------
        X: array, shape = [n_samples, n_features]
            The dataset.

        Returns
        -------
        mapped_X: array, shape = [n_samples, n_landmarks]
            The dataset mapped in the landmarks-based representation.
        """
        mapped_X = []
        for i in range(nb_landmarks):
            transformed_X = transform_cos(Omega[i], X - landmarks_X[i])
            mapped_X.append(np.sum(transformed_X* Q_DA[i], 1))
        return np.array(mapped_X).T
    
    transformed_X_train = pb_mapping_DA(reverse_data['X_S_train'], nb_landmarks, Omega, landmarks_X, Q_DA)
    transformed_X_T_valid = pb_mapping_DA(reverse_data['X_T_valid'], nb_landmarks, Omega, landmarks_X, Q_DA)

    C=param["C"]

    clf = LinearSVC(C=C, random_state=random_state)
    clf.fit(transformed_X_train, reverse_data['y_S_train'])
    t_err = 1 - accuracy_score(reverse_data['y_T_valid'], clf.predict(transformed_X_T_valid))

    return t_err


#Whole learning procedure for PB-landmarks
def pb_landmark(landmarks_X, landmarks_y, reverse_data, param, D, gamma, nb_landmarks,landmarks_selection_method, Omega, random_state):
    ns, d = reverse_data['X_S_train'].shape
    nt, d = reverse_data['X_T_train'].shape

    

    loss = []
    # Computing loss for each landmarks

    #Compute loss per landmark
    for i in range(nb_landmarks):
        transformed_X = transform_cos(Omega[i], reverse_data['X_S_train'] - landmarks_X[i])

        lambda_y = -np.ones(ns)
        lambda_y[(reverse_data['y_S_train'] == landmarks_y[i])] = 1

        landmark_loss = lambda_y @ transformed_X

        # For the clustering method, landmarks are not sampled directly from dataset
        if landmarks_selection_method == "clustering":
            landmark_loss = landmark_loss / (ns)

        # For the random method, case where X_i == landmark needs to be substracted
        elif landmarks_selection_method == "random":
            landmark_loss = (landmark_loss - 1) / (ns - 1)

        landmark_loss = (1 - landmark_loss) / 2
        loss.append(landmark_loss)

    loss = np.array(loss)

    ##COMPUTE Q

    beta = param["beta"]
    # Computing t
    t = beta * sqrt(ns)

    # Computing Q
    Q = -t*loss - logsumexp(-t*loss, axis=1).reshape(-1, 1)
    Q = np.exp(Q)


    def pb_mapping(X):
        """PAC-Bayesian landmarks-based mapping of X according to computed pseudo-posterior Q distribution.

        Parameters
        ----------
        X: array, shape = [n_samples, n_features]
            The dataset.

        Returns
        -------
        mapped_X: array, shape = [n_samples, n_landmarks]
            The dataset mapped in the landmarks-based representation.
        """
        mapped_X = []
        for i in range(nb_landmarks):
            transformed_X = transform_cos(Omega[i], X - landmarks_X[i])
            mapped_X.append(np.sum(transformed_X* Q[i], 1))

        return np.array(mapped_X).T
    
    transformed_X_train = pb_mapping(reverse_data['X_S_train'])
    transformed_X_T_valid = pb_mapping(reverse_data['X_T_valid'])
    

    C=param["C"]
    clf = LinearSVC(C=C, random_state=random_state)
    clf.fit(transformed_X_train, reverse_data['y_S_train'])
    t_err = 1 - accuracy_score(reverse_data['y_T_valid'], clf.predict(transformed_X_T_valid))
    
    return t_err



def transform_cos(omega, delta):
        """Hypothesis computation: h_omega(delta)

        Parameters
        ----------
        omega: array, shape = [d, D]
            omega values (sampled from the Fourier features).

        delta: array, shape = [n, d]
            Pairwise distances.

        Returns
        -------
        hypothesis: array, shape = [n, D]
            Hypothesis values.
        """
        return np.cos(np.dot(delta, omega))



def select_landmarks(X_T_train, y_T_train, nb_landmarks, landmarks_selection_method, random_state):

    n_landmarks_per_label = int(int(nb_landmarks) / len(np.unique(y_T_train)))
    landmarks_selector = LandmarksSelector(n_landmarks_per_label, landmarks_selection_method, random_state=random_state)
    landmarks_X, landmarks_y = landmarks_selector.fit(X_T_train, y_T_train)

    return landmarks_X, landmarks_y







