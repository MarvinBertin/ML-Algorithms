import numpy as np

def logistic_func(x):
    '''
    INPUT: 1 dimensional numpy array, numpy array
    OUTPUT: numpy array
    '''
    return 1. / (1 + np.exp(-x))

def hypothesis(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted percentages (floats between 0 and 1) for the given
    data with the given coefficients.
    '''
    return logistic_func(X.dot(coeffs))

def predict(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted values (0 or 1) for the given data with the given
    coefficients.
    '''
    return (hypothesis(X, coeffs) >= 0.5).astype(int)

def log_likelihood(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: float

    Calculate the log likelihood of the data with the given coefficients.
    '''
    h = hypothesis(X, coeffs)
    return np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

def log_likelihood_gradient(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: numpy array

    Calculate the gradient of the log likelihood at the given value for the
    coeffs. Return an array of the same size as the coeffs array.
    '''
    h = hypothesis(X, coeffs)
    return (y - h).dot(X)

def accuracy(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUPUT: float

    Calculate the percent of predictions which equal the true values.
    '''
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float

    Calculate the percent of positive predictions which were correct.
    '''
    return np.sum((y_true == y_pred)[y_true == 1]) / float(np.sum(y_pred))

def recall(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float

    Calculate the percent of positive cases which were correctly predicted.
    '''
    return np.sum((y_true == y_pred)[y_true == 1]) / float(np.sum(y_true))
