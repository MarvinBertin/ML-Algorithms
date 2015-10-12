import numpy as np


class GradientAscent(object):

    def __init__(self, cost, gradient, predict_func, fit_intercept = True, scale = True):
        '''
        INPUT: GradientAscent, function, function
        OUTPUT: None

        Initialize class variables. Takes two functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        '''
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.costs_ascent = []
        self.fit_intercept = fit_intercept
        self.scale = scale

    def run(self, X, y, alpha=0.01, num_iterations=10000, L2_reg = 0):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None

        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        if self.scale:
            self.scale_fit(X)
            X = self.scale_transform(X)
        
        if self.fit_intercept:
            X = self.add_intercept(X)
            
        self.coeffs = np.zeros(X.shape[1])
        
        print "Starting Gradient Ascent with theta = {} and cost = {}".format(self.coeffs, self.cost(X, y, self.coeffs, L2_reg))
        print "Running..."
        
        for _ in xrange(num_iterations):
            self.coeffs += alpha * self.gradient(X, y, self.coeffs, L2_reg)
            self.costs_ascent.append(self.cost(X, y, self.coeffs, L2_reg))
        
        print "After {} iterations: theta = {} and cost = {}".format(num_iterations, self.coeffs,
                                                                     self.cost(X, y, self.coeffs, L2_reg))
    
    
    def predict(self, X):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)

        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's. Call self.predict_func.
        '''
        if self.scale:
            X = self.scale_transform(X)
            
        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.predict_func(X, self.coeffs)
    
    def add_intercept(self, X):
        '''
        INPUT: 2 dimensional numpy array
        OUTPUT: 2 dimensional numpy array

        Return a new 2d array with a column of ones added as the first
        column of X.
        '''
        return np.vstack((np.ones(len(X)), X.T)).T
    
    def scale_fit(self, X):
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0, ddof=1)
    
    def scale_transform(self, X):
        return (X - self.mean) / self.std
    
#     def probabilities(self, X)
#         X_b = np.vstack((np.ones(len(X)), X.T)).T
#         return self.predict_func(X_b, self.coeffs)
