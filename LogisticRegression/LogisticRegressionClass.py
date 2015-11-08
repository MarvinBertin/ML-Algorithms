import numpy as np

class LogisticReg(object):

    def __init__(self, fit_intercept = True, scale = True, regularizarion = "L2"):
        '''
        INPUT: GradientDescent, function, function, function
        OUTPUT: None

        Initialize class variables. Takes three functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        predict: function to calculate the predicted values (0 or 1) for 
        the given data
        '''
        self.coeffs = None
        self.gamma = 0
        self.costs_descent = []
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.reg = regularizarion

    def fit(self, X, y, alpha=0.01, num_iterations=10000, gamma=0):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None

        Run the gradient descent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        self.gamma = gamma
        
        if self.scale:
            self.scale_fit(X)
            X = self.scale_transform(X)
        
        if self.fit_intercept:
            X = self.add_intercept(X)
            
        self.coeffs = np.random.randn(X.shape[1])
        
        print "Starting Gradient Descent with coefficients = {} and cost = {}".format(
               self.coeffs, self.cost_regularized(X, y))
        print "Running..."
        
        for _ in xrange(num_iterations):
            self.coeffs -= alpha * self.gradient_regularized(X, y)
            self.costs_descent.append(self.cost_regularized(X, y))
        
        print "After {} iterations: theta = {} and cost = {}".format(num_iterations, self.coeffs,
                                                                     self.cost_regularized(X, y))
        
    
    def add_intercept(self, X):
        '''
        INPUT: 2 dimensional numpy array
        OUTPUT: 2 dimensional numpy array

        Return a new 2d array with a column of ones added as the first
        column of X.
        '''        
        return np.insert(X, 0, 1, axis=1)
    
    def hypothesis(self, X):
        '''
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: numpy array

        Calculate the predicted percentages (floats between 0 and 1) 
        for the given data with the given coefficients.
        '''

        return 1 / (1 + np.exp(-X.dot(self.coeffs)))

    def predict(self, X):
        '''
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: numpy array

        Calculate the predicted values (0 or 1) for the given data with 
        the given coefficients.
        '''
        if self.scale:
            X = self.scale_transform(X)
            
        if self.fit_intercept:
            X = self.add_intercept(X)
            
        return (self.hypothesis(X) >= 0.5).astype(int)

    def cost_function(self, X, y):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        Calculate the value of the cost function for the data with the 
        given coefficients.
        '''
        h = self.hypothesis(X)
        return - np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def cost_regularized(self, X, y):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        Calculate the value of the cost function with regularization 
        for the data with the given coefficients.
        '''
        thetas = self.coeffs[1:]
        N = float(X.shape[0])
        
        if self.reg == "L2":
            return self.cost_function(X, y) + self.gamma * np.sum(thetas**2) / (2 * N)
        
        elif self.reg == "L1":
            return self.cost_function(X, y) + self.gamma * np.sum(np.abs(thetas)) / (2 * N)

    def cost_gradient(self, X, y):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function at the given value 
        for the coeffs. 

        Return an array of the same size as the coeffs array.
        '''
        return (self.hypothesis(X) - y).dot(X)

    def gradient_regularized(self, X, y):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function with regularization 
        at the given value for the coeffs. 

        Return an array of the same size as the coeffs array.
        '''
        thetas = self.coeffs.copy()
        thetas[0] = 0
        N = float(X.shape[0])
        
        if self.reg == "L2":
            return self.cost_gradient(X, y) + self.gamma * thetas / N
        
        elif self.reg == "L1":
            thetas[1:] = 1
            return self.cost_gradient(X, y) + self.gamma * thetas / N

    
    def scale_fit(self, X):
        '''
        INPUT: 2 dimensional numpy array
        
        Calculates the mean and standard deviation of the input data
        '''
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
    def scale_transform(self, X):
        '''
        INPUT: 2 dimensional numpy array
        OUTPUT: 2 dimensional numpy array
        
        Transform and normalize a dataset
        '''
        return (X - self.mean) / self.std
    
    def probabilities(self, X):
        '''
        INPUT: 2 dimensional numpy array
        OUTPUT: numpy array
        
        Calculates the predicted probabilities
        '''
        
        if self.scale:
            X = self.scale_transform(X)
            
        if self.fit_intercept:
            X = self.add_intercept(X)
            
        return self.hypothesis(X)
    
    def accuracy(self, y_true, y_pred):
        '''
        INPUT: numpy array, numpy array
        OUPUT: float

        Calculate the percent of predictions which equal the true values.
        '''
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred):
        '''
        INPUT: numpy array, numpy array
        OUTPUT: float

        Calculate the percent of positive predictions which were correct.
        '''
        return np.sum((y_true == y_pred)[y_true == 1]) / float(np.sum(y_pred))
    
    def recall(self, y_true, y_pred):
        '''
        INPUT: numpy array, numpy array
        OUTPUT: float

        Calculate the percent of positive cases which were correctly predicted.
        '''
        return np.sum((y_true == y_pred)[y_true == 1]) / float(np.sum(y_true))
    