from math import ceil, sqrt, exp, log

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

class LandmarksBasedLearner(object):
    """Landmarks-Based learner class

    Parameters
    ----------
    dataset: dict
        The dataset as a dictionnary with the following keys:
        X_train, X_valid, X_test, y_train, y_valid, y_test, name.

    C_range: list
        C values range to search from (SVM's penalty parameter).
        Used while learning a linear classifier over the mapped dataset.

    gamma: float
        Gamma value (RBF kernel's bandwith parameter).
        Used for sampling the Fourier features.

    landmarks_selection_method: str
        The landmarks selection method from: {random, clustering}.

    random_state: None, int or instance of RandomState.
        Information about the random state to be used.


    Attributes
    ----------
    dataset: dict
        The dataset as a dictionnary with the following keys:
        X_train, X_valid, X_test, y_train, y_valid, y_test, name.

    n: int
        Number of samples in the training set (X_train.shape[0]).

    d: int
        Number of features in the dataset (X_train.shape[1]).

    C_range: list
        C values range to search from (SVM's penalty parameter).

    gamma: float
        Gamma value (RBF kernel's bandwith parameter).

    sigma: float
        Sigma value computed using gamma value: sigma = 1 / sqrt(2 * gamma)

    landmarks_selection_method: str
        The landmarks selection method from: {random, clustering}.

    random_state: instance of RandomState.
        Random state for all random operations.

    percentage_landmarks: float
        Percentage of training set samples used as landmarks.

    n_landmarks: int
        Number of landmarks.

    landmarks_X: array, shape = [n_landmarks, d]
        Landmarks.

    landmarks_y: array, shape = [n_landmarks]
        Labels of the landmarks.

    beta: float
        Beta value (pseudo-posterior "temperature" parameter).

    Q: array, shape = [n_landmarks, D]
        Pseudo-posterior distributions over the features (one per landmark).

    D: int
        Number of Fourier features per landmarks.

    Omega: array, shape = [n_landmarks, d, D]
        Feature's omega vector sampled from the Fourier distribution.

    loss: array, shape = [n_landmarks, D]
        Empirical losses matrix (one loss per Fourier feature)
    """

    def __init__(self, dataset, C_range, gamma, landmarks_selection_method, random_state=42):
        self.dataset = dataset
        self.ns, self.d = self.dataset['X_S_train'].shape
        self.nt, self.d = self.dataset['X_T_train'].shape
        self.C_range = C_range
        self.gamma = gamma
        self.sigma = 1. / sqrt(2 * self.gamma)
        self.landmarks_selection_method = landmarks_selection_method
        self.random_state = check_random_state(random_state)

    def select_landmarks(self, nb_landmarks):
        """Select landmarks from a dataset using LandmarksSelector.

        Parameters
        ----------
        percentage_landmarks: float
            Percentage of training set samples to be used as landmarks.

        """
        self.ns_landmarks = nb_landmarks
        n_landmarks_per_label = int(int(nb_landmarks) / len(np.unique(self.dataset['y_S_train'])))
        landmarks_selector = LandmarksSelector(n_landmarks_per_label, self.landmarks_selection_method, random_state=self.random_state)
        self.landmarks_X, self.landmarks_y = landmarks_selector.fit(self.dataset['X_S_train'], self.dataset['y_S_train'])

    def compute_Q(self, beta):
        """Compute pseudo-posterior Q distribution over the Fourier features.

        Parameters
        ----------
        beta: float
            Beta value (pseudo-posterior "temperature" parameter).
        """
        self.beta = beta
        # Computing t
        t = self.beta * sqrt(self.ns)

        # Computing Q
        self.Q = -t*self.loss - logsumexp(-t*self.loss, axis=1).reshape(-1, 1)
        self.Q = np.exp(self.Q)

    def transform_cos(self, omega, delta):
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

    def compute_loss_DA(self, D):
        """Compute loss for a given number of Fourier features per landmarks.

        Parameters
        ----------
        D: int
            Number of Fourier features per landmarks.
        """
        self.D = D

        # Randomly sampling Omega from the Fourier distribution

        self.Omega = self.random_state.randn(self.ns_landmarks, self.d, self.D) / (1. / sqrt(2 * self.gamma))
        loss = []
        # Computing loss for each landmarks
        for i in range(self.ns_landmarks):
            transformed_X = self.transform_cos(self.Omega[i], self.dataset['X_S_train'] - self.landmarks_X[i])

            lambda_y = -np.ones(self.ns)
            lambda_y[(self.dataset['y_S_train'] == self.landmarks_y[i])] = 1

            landmark_loss = lambda_y @ transformed_X

            # For the clustering method, landmarks are not sampled directly from dataset
            if self.landmarks_selection_method == "clustering":
                landmark_loss = landmark_loss / (self.ns)

            # For the random method, case where X_i == landmark needs to be substracted
            elif self.landmarks_selection_method == "random":
                landmark_loss = (landmark_loss - 1) / (self.ns - 1)

            landmark_loss = (1 - landmark_loss) / 2
            loss.append(landmark_loss)
        self.loss_DA = np.array(loss)

    def compute_loss(self, D):
        """Compute loss for a given number of Fourier features per landmarks.

        Parameters
        ----------
        D: int
            Number of Fourier features per landmarks.
        """
        self.D = D

        # Randomly sampling Omega from the Fourier distribution
        self.Omega = self.random_state.randn(self.ns_landmarks, self.d, self.D) / (1. / sqrt(2 * self.gamma))

        loss = []
        # Computing loss for each landmarks
        for i in range(self.ns_landmarks):
            transformed_X = self.transform_cos(self.Omega[i], self.dataset['X_S_train'] - self.landmarks_X[i])

            lambda_y = -np.ones(self.ns)
            lambda_y[(self.dataset['y_S_train'] == self.landmarks_y[i])] = 1

            landmark_loss = lambda_y @ transformed_X

            # For the clustering method, landmarks are not sampled directly from dataset
            if self.landmarks_selection_method == "clustering":
                landmark_loss = landmark_loss / (self.ns)

            # For the random method, case where X_i == landmark needs to be substracted
            elif self.landmarks_selection_method == "random":
                landmark_loss = (landmark_loss - 1) / (self.ns - 1)

            landmark_loss = (1 - landmark_loss) / 2
            loss.append(landmark_loss)
        self.loss = np.array(loss)

     ####### NEW #######

    def compute_Q_DA(self, beta_da, c, b):
        """Compute pseudo-posterior Q distribution over the Fourier features.

        Parameters
        ----------
        beta: float
            Beta value (pseudo-posterior "temperature" parameter).
        """
        c_plus=c/(1-exp(-c))
        b_plus=beta_da*b/(1-exp(-b))
        self.c =c
        self.b =b
        def kl_divergence(q):
            epsilon=0.0001
            q=q+epsilon
            return sum(q[i] * log(q[i]/(1/self.D)) for i in range(len(q)))

        self.beta_da = beta_da
        Bounds(lb=0, ub=1, keep_feasible=False)
        # Computing t
        t = self.beta_da * sqrt(self.ns)
        Q_list=[]
        for i in range(self.ns_landmarks):
            Q=np.repeat(1/self.D, self.D)
            Q=np.array(Q)
            func= lambda x: c_plus * 1/2 * np.transpose(x) @ self.disagreement[i] @ x + b_plus * np.transpose(x) @ self.joint_error[i] @ x + ((c_plus/(self.ns*c))+(b_plus/(self.ns*b)))*2*kl_divergence(x)
            #res=minimize(func, Q, bounds=Bounds(lb=0.0001, ub=1, keep_feasible=False), method='L-BFGS-B')
            res=minimize(func, Q, bounds=Bounds(lb=0.01, ub=1, keep_feasible=False), constraints=LinearConstraint(np.ones(len(Q)), lb=1.0, ub=1.0, keep_feasible=False), method='trust-constr')
            Q=res.x
            Q=Q/sum(Q)
        # Computing Q
            Q_list.append(Q)
        self.Q_DA = Q_list

    ####### NEW #######

    def compute_disagreement(self):
        """Compute loss for a given number of Fourier features per landmarks.

        Parameters
        ----------
        D: int
            Number of Fourier features per landmarks.
        """

        disagreement=[]
        # Computing loss for each landmarks
             
        for i in range(self.ns_landmarks):
            transformed_X = self.transform_cos(self.Omega[i], self.dataset['X_T_train'] - self.landmarks_X[i])

            landmark_disagreement=np.transpose(transformed_X) @ transformed_X

            landmark_disagreement=landmark_disagreement/self.ns

            # For the clustering method, landmarks are not sampled directly from dataset
            if self.landmarks_selection_method == "clustering":
                landmark_disagreement = landmark_disagreement / (self.ns)

            # For the random method, case where X_i == landmark needs to be substracted
            elif self.landmarks_selection_method == "random":
                landmark_disagreement = (landmark_disagreement - 1) / (self.ns - 1)

            landmark_disagreement = (1 - landmark_disagreement) / 2
            disagreement.append(landmark_disagreement)
        self.disagreement = np.array(disagreement)

    ##### NEW #####

    def compute_joint_error(self):
        """Compute loss for a given number of Fourier features per landmarks.

        Parameters
        ----------
        D: int
            Number of Fourier features per landmarks.
        """

        D=self.D
        joint_error = []
        # Computing loss for each landmarks
        for i in range(self.ns_landmarks):
            transformed_X = self.transform_cos(self.Omega[i], self.dataset['X_S_train'] - self.landmarks_X[i])

            lambda_y = -np.ones(self.ns)
            lambda_y[(self.dataset['y_S_train'] == self.landmarks_y[i])] = 1
            to_stack=lambda_y
            #landmark_loss = lambda_y @ transformed_X
            for j in range(D-1):
                lambda_y = np.vstack((lambda_y,to_stack))

            #chaque hypothése multipliée par le lambda associé
            landmark_loss=np.multiply(np.transpose(lambda_y), transformed_X)
            #la loss pour chaque éléments et chaque hypothése
            landmark_loss = (1 - landmark_loss) / 2

            landmark_joint_error= np.transpose(landmark_loss) @ landmark_loss
            # For the clustering method, landmarks are not sampled directly from dataset
            if self.landmarks_selection_method == "clustering":
                landmark_joint_error = landmark_joint_error / (self.ns)

            # For the random method, case where X_i == landmark needs to be substracted
            elif self.landmarks_selection_method == "random":
                landmark_joint_error = (landmark_joint_error - 1) / (self.ns - 1)

            joint_error.append(landmark_joint_error)
        self.joint_error = np.array(joint_error)

    def pb_mapping_DA(self, X):
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
        for i in range(self.ns_landmarks):
            transformed_X = self.transform_cos(self.Omega[i], X - self.landmarks_X[i])
            mapped_X.append(np.sum(transformed_X* self.Q_DA[i], 1))
        return np.array(mapped_X).T
    
    def pb_mapping(self, X):
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
        for i in range(self.ns_landmarks):
            transformed_X = self.transform_cos(self.Omega[i], X - self.landmarks_X[i])
            mapped_X.append(np.sum(transformed_X* self.Q[i], 1))
        return np.array(mapped_X).T
    
    def learn_pb_da(self, C):
        """Learn using PAC-Bayesion landmarks-based mappping.

        Returns
        -------
        results: dict
            Relevant metrics and informations.
        """
        transformed_X_train = self.pb_mapping_DA(self.dataset['X_S_train'])
        transformed_X_S_valid = self.pb_mapping_DA(self.dataset['X_S_valid'])
        transformed_X_T_valid = self.pb_mapping_DA(self.dataset['X_T_valid'])
        transformed_X_test = self.pb_mapping_DA(self.dataset['X_T_test'])

        # C search using a validation set
        
        clf = LinearSVC(C=C, random_state=self.random_state)
        clf.fit(transformed_X_train, self.dataset['y_S_train'])

        val_s_err = 1 - accuracy_score(self.dataset['y_S_valid'], clf.predict(transformed_X_S_valid))
        val_t_err = 1 - accuracy_score(self.dataset['y_T_valid'], clf.predict(transformed_X_T_valid))


        
        # Computing relevant metrics
        mean_max_q = np.mean(np.max(self.Q_DA, axis=1))
        #val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_S_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_T_test'], y_pred)
        f1 = f1_score(self.dataset['y_T_test'], y_pred)

        #plot decision boundary

        """xMin = min(min(transformed_X_train[:, 0]), min(transformed_X_test[:, 0]))-0.1
        xMax = max(max(transformed_X_train[:, 0]), max(transformed_X_test[:, 0]))+0.1
        yMin = min(self.dataset['y_S_train'])-0.1
        yMax = max(max(self.dataset['y_T_train']), max(self.dataset['y_T_test']))+0.1
        xx, yy = np.meshgrid(np.arange(xMin, xMax, .1),
                         np.arange(yMin, yMax, .1))
        mesh = np.c_[xx.ravel(), yy.ravel()]

        rows = 1
        columns = 1
        fig = plt.figure(1, figsize=(columns*10, rows*8))

        c1 = "#3C4EC2"
        c2 = "#B50927"

        ax = fig.add_subplot(rows, columns,1)
        ax.scatter(transformed_X_train[:, 0], transformed_X_train[:, 1], c=self.dataset['y_S_train'], s=[500]*len(transformed_X_train), marker="o", lw=0, edgecolors='black', cmap=ListedColormap([c1, c2]))
        ax.scatter(transformed_X_test[:, 0], transformed_X_test[:, 1], c=self.dataset['y_T_test'], lw=1,
               s=[500]*len(transformed_X_test), marker="P", edgecolors='black', cmap=ListedColormap([c1, c2]))
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        minZ = np.min(Z)
        maxZ = np.max(Z)
        stepZ = (maxZ-minZ) / 10
        levels = np.arange(minZ, maxZ+stepZ, stepZ)
        ctrf = ax.contourf(xx, yy, Z, levels, cmap=plt.cm.RdBu, zorder=0)

        ax.contour(xx, yy, Z, levels, cmap=plt.cm.RdBu, zorder=0)
        ax.contour(xx, yy, Z, [0], colors=["purple"], zorder=3,
               linestyles=["solid"], linewidths=[4])
        fig.colorbar(ctrf, orientation='vertical')
        fig.savefig("perc_l"+ str(self.percentage_landmarks) + "Beta" + str(self.beta) + "b" + str(self.b) + "c" + str(self.c) + ".pdf", bbox_inches="tight")
        print("saved!!!")"""

        return dict([("dataset", self.dataset['name']), ("exp", 'landmarks'), ("algo", 'PBDA'), ("method", self.landmarks_selection_method), \
                    ("C", C), ("D", self.D), ("c", self.c),("b", self.b), ("n_landmarks", self.ns_landmarks), \
                    ("gamma", self.gamma), ("beta_da", self.beta_da), ("train_error", train_err), ("val_s_error", val_s_err), ("val_t_error", val_t_err), ("test_error", test_err), \
                    ("f1", f1), ("mean_max_q", mean_max_q)])

    def learn_pb(self, C):
        """Learn using PAC-Bayesion landmarks-based mappping.

        Returns
        -------
        results: dict
            Relevant metrics and informations.
        """
        transformed_X_train = self.pb_mapping(self.dataset['X_S_train'])
        transformed_X_S_valid = self.pb_mapping(self.dataset['X_S_valid'])
        transformed_X_T_valid = self.pb_mapping(self.dataset['X_T_valid'])
        transformed_X_test = self.pb_mapping(self.dataset['X_T_test'])

        # C search using a validation set

        clf = LinearSVC(C=C, random_state=self.random_state)
        clf.fit(transformed_X_train, self.dataset['y_S_train'])
       
        val_s_err = 1 - accuracy_score(self.dataset['y_S_valid'], clf.predict(transformed_X_S_valid))
        val_t_err = 1 - accuracy_score(self.dataset['y_T_valid'], clf.predict(transformed_X_T_valid))

            

        # Computing relevant metrics
        mean_max_q = np.mean(np.max(self.Q, axis=1))
        #val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_S_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_T_test'], y_pred)
        f1 = f1_score(self.dataset['y_T_test'], y_pred)

        return dict([("dataset", self.dataset['name']), ("exp", 'landmarks'), ("algo", 'PB'), ("method", self.landmarks_selection_method), \
                    ("C", C), ("D", self.D), ("n_landmarks", self.ns_landmarks), \
                    ("gamma", self.gamma), ("beta", self.beta), ("train_error", train_err), ("val_s_error", val_s_err), ("val_t_error", val_t_err), ("test_error", test_err), \
                    ("f1", f1), ("mean_max_q", mean_max_q)])
    

    def rbf_mapping(self, X):
        """RBF landmarks-based mapping of X.

        Parameters
        ----------
        X: array, shape = [n_samples, n_features]
            The dataset.

        Returns
        -------
        mapped_X: array, shape = [n_samples, n_landmarks]
            The dataset mapped in the landmarks-based representation.
        """
        return np.exp(-self.gamma * cdist(X, self.landmarks_X, 'sqeuclidean'))

    def learn_rbf(self, C):
        """Learn a linear SVM over the PAC-Bayesian landmarks-based mapping.

        Returns
        -------
        results: dict
            Relevant metrics and informations.
        """
        transformed_X_train = self.rbf_mapping(self.dataset['X_S_train'])
        transformed_X_S_valid = self.rbf_mapping(self.dataset['X_S_valid'])
        transformed_X_T_valid = self.rbf_mapping(self.dataset['X_T_valid'])
        transformed_X_test = self.rbf_mapping(self.dataset['X_T_test'])

        clf = LinearSVC(C=C, random_state=self.random_state)
        clf.fit(transformed_X_train, self.dataset['y_S_train'])
        val_s_err = 1 - accuracy_score(self.dataset['y_S_valid'], clf.predict(transformed_X_S_valid))
        val_t_err = 1 - accuracy_score(self.dataset['y_T_valid'], clf.predict(transformed_X_T_valid))

        # Computing relevant metrics
        train_err = 1 - accuracy_score(self.dataset['y_S_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_T_test'], y_pred)
        f1 = f1_score(self.dataset['y_T_test'], y_pred)

        return dict([("dataset", self.dataset['name']), ("exp", 'landmarks'), ("algo", 'RBF'), ("method", self.landmarks_selection_method), \
                    ("C", C), ("n_landmarks", self.ns_landmarks), ("gamma", self.gamma),\
                    ("train_error", train_err), ("val_s_error", val_s_err), ("val_t_error", val_t_err),("test_error", test_err), ("f1", f1)])


def compute_landmarks_selection(args, dataset, C_range, gamma, random_state):
    """Landmarks selection function for parallel processing."""
    landmarks_based_learner = LandmarksBasedLearner(dataset, C_range, gamma, args['method'], random_state)
    landmarks_based_learner.select_landmarks(args['nb_landmarks'])

    print(f"Processing: {args['nb_landmarks']} landmarks {args['method']} selection")
    with open(args['output_file'], 'wb') as out_file:
        pickle.dump(landmarks_based_learner, out_file, protocol=4)

    return args['method']

def compute_landmarks_based(args, beta_range):
    """Landmarks-based learning function for parallel processing."""
    tmp_results = []

    with open(args["input_file"], 'rb') as in_file:
        landmarks_based_learner = pickle.load(in_file)

    if args["algo"] == "rbf":
        print(f"Processing: rff with {args['nb_landmarks']} landmarks {args['method']}")
        tmp_results.append(landmarks_based_learner.learn_rbf())

    elif args["algo"] == "pb":
        print(f"Processing: pb with {args['D']} features, {args['nb_landmarks']} landmarks {args['method']}")
        landmarks_based_learner.compute_loss(args['D'])
        for beta in beta_range:
            landmarks_based_learner.compute_Q(beta)
            tmp_results.append(landmarks_based_learner.learn_pb())

    with open(args["output_file"], 'wb') as out_file:
        pickle.dump(tmp_results, out_file, protocol=4)

    return args["algo"]

def compute_landmarks_based_DA(args, beta_range, beta_DA_range, c_range, b_range, C_range):
    """Landmarks-based learning function for parallel processing."""
    tmp_results = []

    with open(args["input_file"], 'rb') as in_file:
        landmarks_based_learner = pickle.load(in_file)

    if args["algo"] == "rbf":
        print(f"Processing: rff with {args['nb_landmarks']} landmarks {args['method']}")
        for C in C_range:
            tmp_results.append(landmarks_based_learner.learn_rbf(C))

    elif args["algo"] == "pb_da":
        print(f"Processing: pb_da with {args['D']} features, {args['nb_landmarks']} landmarks {args['method']}")
        landmarks_based_learner.compute_loss_DA(args['D'])
        #### NEW ####
        #attention à etre certain de compute_loss avant pour avoir les omegas
        landmarks_based_learner.compute_disagreement()
        landmarks_based_learner.compute_joint_error()
        for beta_da in beta_DA_range:
            for c in c_range:
                for b in b_range:
                    for C in C_range:
                        landmarks_based_learner.compute_Q_DA(beta_da=beta_da, c=c, b=b)
                        tmp_results.append(landmarks_based_learner.learn_pb_da(C))

    elif args["algo"] == "pb":
        print(f"Processing: pb with {args['D']} features, {args['nb_landmarks']} landmarks {args['method']}")
        landmarks_based_learner.compute_loss(args['D'])
        for beta in beta_range:
            for C in C_range:
                landmarks_based_learner.compute_Q(beta)
                tmp_results.append(landmarks_based_learner.learn_pb(C))

    #pas DA, meme que la fonction de base mais la loss est deja calculée
    
    

    with open(args["output_file"], 'wb') as out_file:
        pickle.dump(tmp_results, out_file, protocol=4)

    return args["algo"]