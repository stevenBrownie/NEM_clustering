import numpy as np
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import itertools


"""A standard GMM EM algoithm has been used as template. 

Gaussian Mixture Model EM Algorithm - Vectorized implementation
Xavier Bourret Sicotte, 2018
https://github.com/xavierbourretsicotte/xavierbourretsicotte.github.io/blob/5039e68f3e0cebbe58552adf7ac4b479117a6671/downloads/notebooks/EM_Gaussian_Mixture_Model.ipynb

Then the code has been adapted for 1D NEM."""

class NEM_GaussianMixture(object):
    """Gaussian Mixture Model - 1D implementation 
    with NEM algorithm"""
    
    def __init__(self, n_components = 3, max_iter = 600, tol = 0.001,beta=0.1):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.variances_ = None
        self.U = None
        self.beta=beta
        self.W_=None
        self.convergence=True

    def init_kmeans(self,X):
        """K-means algorithm for initialisation of gaussian parameters
        Args:
        X: (n,1) n x 1 dimensional data
        Returns:
        mu_km: vector with initialised mean for each gaussian component
        Sigma_km: vector with initialised variance  for each gaussian component
        """

        #use k-means to initilaise mu and sigma:
        km = KMeans(n_clusters=self.n_components,n_init="auto").fit(X)
        mu_km=km.cluster_centers_
        Sigma_km=np.array([])

        y_km=km.labels_
        labels=np.unique(y_km)
        for l in labels:
             #get cluster indices
             cluster_indices=np.where(y_km== l)[0]
             cluster=X[cluster_indices]#get the values in cluster
             Sigma_km=np.append(Sigma_km,np.var(cluster))#add var
        return mu_km,Sigma_km
    
    def fit(self,X,V,kmeans_init=False):
        """
        This function tries to fit gaussian components to data using NEM if beta>0 or EM if beta = 0.
        Args:
            kmeans_init: (bool) If the kmeans is used to initialise.
        """
        n,d = X.shape  # n = datapoints, d = features
        k = self.n_components  #K number of clusters

        #initilise the gaussian's mean and variance

        #use kmeans
        if kmeans_init==True:
             mu,Sigma=self.init_kmeans(X)
        #random init
        else:
            # randomly initialize the starting means
            np.random.seed(1) #seed is important for reproducabilty 
            mu = X[np.random.choice(n,k,replace = False)]

            # initialize a covariance matrix for each gaussian: diagonal matrix
            Sigma = [np.eye(d)] * k

        # initialize the probability for each gaussian pi
        pi = np.array([1 / k] * k)

        # initialize responsibility matrix: n points for each gaussian
        W = np.zeros((n,k))

        # initialize list of log-likelihoods
        U = []
       #===============================================================#
    
        while len(U) < self.max_iter:
            ########## Expectation step#########
            W,W_res=self.fixed_point(W,V,pi,mu,Sigma, self.beta,k,n,X)

            # Sum of loglikelihood and spatial factor: U is the function we maximise
            G=self.G_reg(W,V) #spatial factor
            l= np.sum(np.log(np.sum(W_res, axis = 1)))#loglikelihood
            if self.beta!=0:#use NEM
                u = l+G
            else:#normal EM algorithm
                u=l
            # store log likelihood in list
            U.append(u)

            # sum of w^i entries along j (used for parameter updates)
            W_s = np.sum(W, axis = 0)

            ########## Maximisation step##########
            for j in range(k):

                ## Update means
                mu[j] = (1. / W_s[j]) * np.sum(W[:, j] * X.T, axis = 1).T
                ## Update covariances (in 1D only variance)
                var=((W[:,j] * ((X - mu[j]).T)) @ (X - mu[j])) / W_s[j]
                Sigma[j] =var
                if Sigma[j]==0:#Add regularization to avoid singualarity: a normal pdf with sigma = 0 is not allowed. This happens if cluster size =1.
                    Sigma[j]=1e-08
                pi[j] = W_s[j] / n

            # check for convergence
            if len(U) < 2: continue
            if np.abs(u - U[-2]) < self.tol: break
        
        if np.abs(u - U[-2]) > self.tol:
            print("**** ERROR DID NOT converge ****")
            self.convergence=False

        self.means_ = mu
        self.variances_ = Sigma
        self.U = U
        self.W_=W
        
    def G_reg(self,W,V):
        """
        This function calculates the spatial part G of the criterion U.
        Args:
            W is classifcation matrix.
            V neighbourhood matrix
        Returns:
        G(W,V)
        """
        n,d = W.shape  ## n = datapoints, d = features
        G=0
        for i in range(n):
            for j in range(i,n):
                G+=V[i][j]*W[i].T@W[j]
        return G
    def fixed_point(self,W,V,pi,mu,Sigma,beta,k,n,X,max_iterations=1000,float_type=np.float64,tolerance=1e-3):
        """This function updates the weight matrix W (classifcation matrix). To calculate the matrix an approximisation based on Fixed-point iteration is used.
        Large exponetials are computed during this step thus it is advised increasing the precision of floats to 32 or 64 bits. 
        
        Args:
            W: The weight matrix containing for each sample the probability of belonging to a certain gaussian.
            V: The neighbour matrix. It contains 1 if ij are neighbors or 0 if they aren't.
            pi: Vector of the probabilty of each gaussian for the mixture model.
            mu: Vector containing the mean for each gaussian
            Sigma: Vector containg the variance for each gaussian.
            K: (int) Number of gaussian
            n: (int) Number of samples
            max_iterations: max number of iterations used to find the solution using the fixed-point iteration method.
            float_type: (dtype) precision of the floats. Increase to avoid overflow errors (they happen for large beta).
            tolerance: (float) precision needed to stop iteration...

        Returns:
           W_new: The update weight matrix (classifcation matrix)
           W_res: The nominator of the weight matrix, used to calculate the loglikelihood.
        """
        W_new=np.array(W,dtype=float_type) #this is important to avoid the overflow: increase precision
        it=0
        # lambda function for gaussian pdf mean m and variance s
        P = lambda m ,s: multivariate_normal.pdf(X, mean = m, cov = s)
        while(it <= max_iterations):
            it=it+1
            W_old=W_new.copy()
            for j in range(k):
                multivariate=P(mu[j], Sigma[j])
                for i in range(n):
                    exponent=beta*(V[i,:]@W_old[:,j])
                    W_new[i, j] = pi[j] *multivariate[i]*(np.exp(exponent))#this can cause an error if exp is too big! caution
            W_res=W_new.copy()
            W_new = (W_new.T / np.exp(np.log(W_new.sum(axis = 1)))).T
            if np.linalg.norm(W_new - W_old) < tolerance:#this line adds a verifcation that there is convergence of the fixed point. This however does prove it is unique.
                return W_new, W_res
        raise ValueError('Solution did not converge') # if didn't converge
    
    def print(self):
        print("Mean = "+ str(self.means_))
        print("Variances = "+ str(self.variances_))
    def fit_predict(self):
        """This function classifies each point to the most probable component.
        Returns:
            vector containing the class for each data point."""
        return np.argmax(self.W_, axis = 1)
    def plot_U(self):
        """Plot the evolution of the criterion U"""
        fig = plt.figure()
        plt.plot(np.arange(len(self.U)),self.U)
        plt.show()

def error_n_body(y_predict,y_truth,N):
    """This function calculates the error between a predicted label and ground truth.
    Since the clusters can classify with different numbers,
    first it reorders the labels and selects the minimum error over the permutations."""
  
    L=len(y_predict)
    y_predict_reordered=np.zeros(L)
    #ordered labels
    label_values=np.linspace(0,N-1,N,dtype=int)
    labels_perm=list(itertools.permutations(label_values))
    L_perm=len(labels_perm)
    err=np.zeros(L_perm)
    for j in range(L_perm):
        for i in range(len(label_values)):
            #switch labeling
            labels=labels_perm[j]
            mask=(y_predict==label_values[i])
            y_predict_reordered[mask==True]=labels[i]
        #add error to list
        for i in range(L):
            #reordered
            if (y_predict_reordered[i] !=y_truth[i]):
                err[j]=err[j]+1

    min_err=np.min(err)
    return min_err*100/L