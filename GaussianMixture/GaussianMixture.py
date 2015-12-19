import numpy as np
import sklearn
from sklearn.datasets import make_classification
from scipy.stats import multivariate_normal 
import pylab
%matplotlib inline


def GaussianMixture(object):
    def __init__(self, k, initial_means, initial_covs):
        self.k = 2
        self.means = initial_means
        self.covs = initial_covs
        self.norm = [None]*self.k
        self._update_norms()
        self.weights = None
        self._alpha = 0.5
        self.X = None
        
    def _update_norms(self):
        #this fits each norm to the current mean and covariance for that norm
        for index in range(self.k):
            self.norm[index] = multivariate_normal(mean = self.means[index], 
                                                    cov = self.covs[index])
            
    def _expectation(self):
        
        # For all the data points in self.X, calculate the weighting of that datapoint using the _weight method defined
        # below 
        
        #Each x in self.X is assigned a weight based on the equation
        
        
        pass
    
    def _maximization(self):
        
        # Now calculate the weighting means and variance for each of the norms as given to you in the notes. 
        # Make sure to correctly calculate the weighting. Since there are only two norms, what does that mean?
        
        
        pass
    
    
    def _weight(self):
        # Use self.norm[index].pdf(self.x) to calculate the phi as per the equation given to you in the notes. 
        # also use self.alpha
        
        
        
        
        
        pass
    
    def _alpha():
        
        #calculate alpha by summation as given to you in the notes and save to self.alpha
        
        pass
    
    
    def train(self, X, threshold = .001):
        
        # Use the EM algorithm and the various methods you've implemented above to calculate the fit
        
        self.X = X
        
        # You will need to initialize the size of the weights list
        
        # Produce guesses at initial means and variances. These now get passed into the expectation step.
        
        # You want to use a while loop here and keep track of the number of loops the algorithm iterates through
        
        # Make sure to update self._alpha
        
        # If self.ll < threshold, exit the loop
        
        pass
    
    
    def log_likelihood(self):
        
        ll = 0
        for i, x in enumerate(self.X):
            ll += (1-weights[i])*np.log(x)+weights[i]*np.log(x)+\
            (1-weights[i])*np.log(self._alpha)+weights[i]*np.log(1.-self._alpha)
    
        self.ll = ll
        return ll
    
    def predict(self, X):
        
        # Assuming that you have correctly fit the data, given a new datapoint x, how would you 
        # return a label showing which of the two norms x belongs to?
        
        
        #Returns a list of labels of length X.shape[0]
        pass
        
        