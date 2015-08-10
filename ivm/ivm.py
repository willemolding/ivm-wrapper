import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from mlabwrap import mlab

class IVM(BaseEstimator, ClassifierMixin):
    """Import Vector Machine Classifier

    The Import Vector Machines (Zhu and Hastie 2005) are a sparse, discriminative and probabilistic classifier. 
    The algorithm is based on the Kernel Logistic Regression model, but uses only a few data points to define 
    the decision hyperplane in the feature space. These data points are called import vectors.

    The Import Vector Machine shows similar results to the widely used 
    Support Vector Machine, but is inherently multiclass and has a probabilistic output.

    Parameters
    ----------
    sigma : float, optional
        Kernel paramter

    _lambda : float, optional
        regularization paramter

    Nadd : int or inf, optional
        maximum number of points tested for adding to the subset (inf takes all points)

    output : int (0 or 1), optional
        display output (0: no output, 1: output)

    maxIter : int or inf, optional
        maximum number of iterations (maximum number of import vectors, inf tests all points)

    epsilon : float, optional
        stopping criterion for convergence proof

    delta_k : float, optional
        interval for computing the ratio of the negative loglikelihood

    tempInt : float, optional
        temporal filtering, 1: no filtering, 0.5: average between old and new
        parameters (params.tempInit < 1 is more stable, but converges slower)

    epsilonBack : float, optional
        threshold on the function value for backward selection, if an import 
        vector is removed

    flyComputeK : int (1 or 0), optional
        compute kernel on the fly (use, if kernel is too large to compute it at once

    deselect : int (1 or 0), optional
        skip backward selection of import vector (computional cost descrease significantly)

    Attributes
    ----------

    import_ : array-like, shape = [n_IV]
        Indices of import vectors

    n_import_ : int
        Number of support vectors

    alpha_ : array-like
        parameters of the decision hyperplane

    References
    ----------

    .. [1] `Ji Zhu and Trevor Hastie (2005). "Kernel Logistic Regression and the Import Vector Machine". 
       Journal of Computational and Graphical Statistics  Vol. 14(1), pp. 185-205.
        <http://pubs.amstat.org/doi/abs/10.1198/106186005X25619>`_

    .. [2] `Ribana Roscher and Wolfgang Forstner and Bjorn Waske (2012). 
       "I2VM: Incremental import vector machines". Image and Vision Computing Vol. 30(4-5), pp. 263-278.
        <http://www.sciencedirect.com/science/article/pii/S0262885612000546>`_

    """

    def __init__(self, sigma = 0.2, _lambda = np.exp(-15), Nadd = 150, output = 0, maxIter = np.inf,
                epsilon = 0.001, delta_k = 1, tempInt = 0.95, epsilon_back = 0.001, flyComputeK = 0,
                deselect = 0, CV = None, lambdas = np.r_[[np.exp(-12)], np.exp(np.arange(-10,-3))], 
                sigmas = np.sqrt(1. / (2. * 2.**np.arange(-5, 3)))):

        self.sigma = sigma
        self._lambda = _lambda
        self.Nadd = Nadd
        self.output = output
        self.maxIter = maxIter
        self.epsilon = epsilon
        self.delta_k = delta_k
        self.tempInt = tempInt
        self.epsilon_back = epsilon_back
        self.flyComputeK = flyComputeK
        self.deselect = deselect
        self.CV = CV
        self.lambdas = lambdas
        self.sigmas = sigmas


        this_dir, this_filename = os.path.split(__file__)
        src_dir = os.path.join(this_dir, "ivmSoftware4.3/src")

        mlab.addpath(src_dir)

    def fit(self, X, y):
        """
        Fits the IVM

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns self
        """

        self.classes_, y = np.unique(y, return_inverse=True)
        y = y + 1 #convert to matlab style indexing

        if self.CV is None:
            param_struct = mlab.struct('sigma', self.sigma, 'lambda', self._lambda, 'Nadd', self.Nadd, 'output', self.output, 
                'maxIter', self.maxIter, 'epsilon', self.epsilon, 'delta_k', self.delta_k, 'tempInt', self.tempInt,
                'epsilon_back', self.epsilon_back, 'flyComputeK', self.flyComputeK, 'deselect', self.deselect)

            self.model = mlab.ivm_learn(X.T, y, param_struct)
        else:
            param_struct = mlab.struct('sigmas', np.array(self.sigmas).reshape((1,-1)), 'lambdas', np.array(self.lambdas).reshape((1,-1)), 'CV', self.CV, 'Nadd', self.Nadd, 'output', self.output, 
            'maxIter', self.maxIter, 'epsilon', self.epsilon, 'delta_k', self.delta_k, 'tempInt', self.tempInt,
            'epsilon_back', self.epsilon_back, 'flyComputeK', self.flyComputeK, 'deselect', self.deselect)

            data_struct = mlab.struct('phi', X.T, 'c', y, 'phit', X.T, 'ct', y)

            result = mlab.ivm(data_struct, param_struct)
            self.model = result.model

        return self


    def predict_proba(self, X):
        result = mlab.ivm_predict(self.model, X.T, np.ones(X.shape[0])) #this may be inefficient
        probs = np.array(result.P).T
        return probs

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]