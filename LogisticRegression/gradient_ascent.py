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
        self.costs_ascent = []

    def run(self, X, y, alpha=0.01, num_iterations=10000, reg = None):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None

        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        X_b = np.vstack((np.ones(len(X)), X.T)).T
        self.coeffs = np.zeros(X_b.shape[1])
        #self.coeffs[0] = 200
        
        print "Starting Gradient Ascent with theta = {} and cost = {}".format(self.coeffs, self.cost(X_b, y, self.coeffs, reg))
        print "Running..."
        
        N = float(X_b.shape[0])
        for _ in xrange(num_iterations):
            if reg != None:
                self.coeffs += alpha * (self.gradient(X_b, y, self.coeffs, reg) / N)
                self.costs_ascent.append(self.cost(X_b, y, self.coeffs, reg))
            else:
                self.coeffs += alpha * (self.gradient(X_b, y, self.coeffs) / N)
                self.costs_ascent.append(self.cost(X_b, y, self.coeffs))
        
        print "After {} iterations: theta = {} and cost = {}".format(num_iterations, self.coeffs, self.cost(X_b, y, self.coeffs, reg))
    
    
    def predict(self, X):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)

        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's. Call self.predict_func.
        '''
        X_b = np.vstack((np.ones(len(X)), X.T)).T
        return self.predict_func(X_b, self.coeffs)
    
#     def probabilities(self, X)
#         X_b = np.vstack((np.ones(len(X)), X.T)).T
#         return self.predict_func(X_b, self.coeffs)
