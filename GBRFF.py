import numpy as np
from sklearn.datasets import make_moons
import random
import math
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import fminbound
from matplotlib.colors import ListedColormap


class gbrff(object):
    def __init__(self, gamma=0.1, Lambda=0, T=100, randomState=np.random):
        self.T = T
        self.randomState = randomState
        self.Lambda = Lambda
        self.gamma = gamma

    def loss_grad(self, omega):
        dots = np.dot(omega, self.XT) - self.b
        dott = np.dot(omega, self.XtT) - self.b
        self.yTildePred = np.cos(dots)
        self.yTildePred_t = np.cos(dott)

        v0 = np.exp(self.yTildeN*self.yTildePred)
        return ((1/self.n_s)*np.sum(v0) + self.Lambda*((1/self.n_s)*np.sum(self.yTildePred) 
            - (1/self.n_t)*np.sum(self.yTildePred_t))**2,
                (1/self.n_s)*(self.yTilde*v0*np.sin(dots)).dot(
                    self.X) + self.Lambda*2*((1/self.n_s)*np.sum(self.yTildePred) - (1/self.n_t)*np.sum(self.yTildePred_t))
                    * ((1/self.n_s)*np.sin(dots).dot(self.X) - (1/self.n_t)*np.sin(dott).dot(self.Xt)))

    def fit(self, y, Xs, Xt):
        # Assuming there is two labels in y. Convert them in -1 and 1 labels.
        labels = sorted(np.unique(y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(Xs.shape[0])  # Set all labels at 1
        newY[y == labels[0]] = -1  # except the smallest label in y at -1.
        y = newY
        self.n_s, d = Xs.shape
        self.n_t = Xt.shape[0]
        meanY = np.mean(y)
        self.initPred = 0.5*np.log((1+meanY)/(1-meanY))
        curPred = np.full(self.n_s, self.initPred)
        pi2 = np.pi*2
        self.omegas = np.empty((self.T, d))
        self.alphas = np.empty(self.T)
        self.xts = np.empty(self.T)
        inits = self.randomState.randn(self.T, d)*(2*self.gamma)**0.5
        self.X = Xs
        self.Xt = Xt
        self.XT = Xs.T
        self.XtT = Xt.T
        for t in range(self.T):
            init = inits[t]
            wx_s = init.dot(self.XT)
            wx_t = init.dot(self.XtT)
            w = np.exp(-y*curPred)
            self.yTilde = y*w
            self.yTildeN = -self.yTilde
            self.b = pi2*fminbound(lambda n: (1/self.n_s)*np.sum(np.exp(
                       self.yTildeN*np.cos(pi2*n - wx_s))) + ((1/self.n_s)*np.sum(np.cos(pi2*n - wx_s)) - (1/self.n_t)*np.sum(np.cos(pi2*n - wx_t)))**2, -0.5, 0.5, xtol=1e-2)
            self.xts[t] = self.b
            self.omegas[t], _, _ = optimize.fmin_l_bfgs_b(
                                      func=self.loss_grad, x0=init, maxiter=10)
            vi = (y*self.yTildePred).dot(w)
            vj = np.sum(w)
            alpha = 0.5*np.log((vj+vi)/(vj-vi))
            curPred += alpha*self.yTildePred
            self.alphas[t] = alpha

    def predict(self, X):
        pred = self.initPred+self.alphas.dot(
                                np.cos(self.xts[:, None]-self.omegas.dot(X.T)))
        # Then convert back the labels -1 and 1 to the labels given in fit
        yPred = np.full(X.shape[0], self.positiveLabel)
        yPred[pred < 0] = self.negativeLabel
        return yPred

    def decision_function(self, X):
        return self.initPred+self.alphas.dot(
                                np.cos(self.xts[:, None]-self.omegas.dot(X.T)))


n_samples = 300
seed = 1
np.random.seed(seed)
Xs, ys = make_moons(n_samples, noise=0.05, random_state=seed)
Xt, yt = make_moons(n_samples, noise=0.05, random_state=seed+1)
ys[ys == 0] = -1
yt[yt == 0] = -1
trans = -np.mean(Xs, axis=0)
Xs = 2 * (Xs + trans)
Xt = 2 * (Xt + trans)
degrees = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
#degrees = [30]
rows = len(degrees)
columns = 1
fig = plt.figure(1, figsize=(columns*10, rows*8))
subplotNumber = 0
for degree in degrees:
    theta = -degree*math.pi/180
    rotation = np.array([[math.cos(theta), math.sin(theta)],
                        [-math.sin(theta), math.cos(theta)]])
    Xt_degree = np.dot(Xt, rotation.T)
    da = gbrff(T=100, Lambda=0, gamma=1/Xs.shape[1],
               randomState=np.random.RandomState(1))
    da.fit(ys, Xs, Xt_degree)
    xMin = min(min(Xs[:, 0]), min(Xt_degree[:, 0]))-0.1
    xMax = max(max(Xs[:, 0]), max(Xt_degree[:, 0]))+0.1
    yMin = min(min(Xs[:, 1]), min(Xt_degree[:, 1]))-0.1
    yMax = max(max(Xs[:, 1]), max(Xt_degree[:, 1]))+0.1
    xx, yy = np.meshgrid(np.arange(xMin, xMax, .1),
                         np.arange(yMin, yMax, .1))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    c1 = "#3C4EC2"
    c2 = "#B50927"
    subplotNumber += 1
    ax = fig.add_subplot(rows, columns, subplotNumber)
    ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, s=[500]*len(Xs), marker="o",
               lw=0, edgecolors='black', cmap=ListedColormap([c1, c2]))
    ax.scatter(Xt_degree[:, 0], Xt_degree[:, 1], c=yt, lw=1,
               s=[500]*len(Xt_degree), marker="P", edgecolors='black', cmap=ListedColormap([c1, c2]))
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    Z = -da.decision_function(mesh).reshape(xx.shape)
    minZ = np.min(Z)
    maxZ = np.max(Z)
    stepZ = (maxZ-minZ) / 10
    levels = np.arange(minZ, maxZ+stepZ, stepZ)
    ctrf = ax.contourf(xx, yy, Z, levels, cmap=plt.cm.RdBu, zorder=0)
    ax.contour(xx, yy, Z, levels, cmap=plt.cm.RdBu, zorder=0)
    ax.contour(xx, yy, Z, [0], colors=["purple"], zorder=3,
               linestyles=["solid"], linewidths=[4])
    fig.colorbar(ctrf, orientation='vertical')
    allypredd_s= da.predict(Xs)
    allypredd_t = da.predict(Xt_degree)
    accuracy_s = sum(ys == allypredd_s) / len(ys)
    accuracy_t = sum(yt == allypredd_t) / len(yt)
    #v = "\nmean Xs-Xt " + str( (np.mean(da.decision_function(Xs)) - np.mean(da.decision_function(Xt_degree)))**2)
    #title = ("Degree " + str(degree) +
    #         " Accuracy source: {:5.2f}".format(100*accuracy_s)+ "\n Accuracy target: {:5.2f}".format(100*accuracy_t) + v)
    #ax.set_title(title, fontsize=26)
    ax.set_xticks([])
    ax.set_yticks([])
    #print(title)
fig.subplots_adjust(wspace=0.00, hspace=0.30)
fig.savefig("lambda_equal_"+ str(da.Lambda) +".pdf", bbox_inches="tight")
# Free ram usage with these 4 last lines
for ax in fig.axes:
    ax.cla()
fig.clf()
plt.close(fig)





# On part d'un discriminateur source cible qui assigne - 1 aux sources et +1 aux cibles.
# On cherche à apprendre une représentation des sources uniquement qui maximise la valeur du dsicriminateur sur les sources

# La représentation f n'agit que sur les sources
# On se donne un discriminateur d : source cible

# Etape 1 : on apprend d de façon à discriminer source-cible : GBRFF(X_s,X_t) --> GBRFF_t
# Etape 2 : on apprend f de façon à dégrader les performances de h_t  : max_f (GBRRF_t(f(X_s)))

# On peut commencer par une matrice M quelconque --> max_f (GBRRF_t(M^T X_s))) --> max (cos(omega.(MX_s  +T) - b))


# trois Matrices : R = Rotation --> matrice de rotation en dimension d, i.e. R^T R = I_d
#                  S = Scaling --> matrice diagonale de dimension d \in \mathbb{R} ou même un vecteur 
#                  T = transalation --> vecteur de translation  

# f(X_s) =  R*diag(S) X_s + T

# Contraintes :                

