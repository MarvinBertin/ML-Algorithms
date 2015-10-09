import numpy as np


class GradientAscent(object):

    def __init__(self, cost, gradient, predict_func):
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

    def run(self, X, y, alpha=0.01, num_iterations=10000):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None

        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        self.coeffs = (np.random.rand(X.shape[1]) * 2) - 1
        
        print "Starting Gradient Ascent with theta = {} and cost = {}".format(self.coeffs, self.cost(X, y, self.coeffs))
        print "Running..."
        
        for _ in xrange(num_iterations):
            self.coeffs += alpha * self.gradient(X, y, self.coeffs)
        
        print "After {} iterations: theta = {} and cost = {}".format(num_iterations, self.coeffs, self.cost(X, y, self.coeffs))
    
    
    def predict(self, X):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)

        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's. Call self.predict_func.
        '''
        return self.predict_func(X, self.coeffs)
