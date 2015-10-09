import numpy as np

"""
Gradient Descent:
Vectorirized implementation for a linear regression
"""

def error_vec(theta, X_b, y):
    return y - X_b.dot(theta.T)

def fit_error_vec(theta, X_b, y):
    return sum((error_vec(theta, X_b, y))**2) / float(X_b.shape[0])

def step_gradient_vec(theta, X_b, y, gamma):
    N = float(X_b.shape[0])
    theta_new = theta + (gamma * np.sum((error_vec(theta, X_b, y) * X_b.T), 1) / N)
    return theta_new

def gradient_descent_runner_vec(X_b, y, theta_0, learning_rate, num_iterations):
    theta = theta_0.copy()
    for i in range(num_iterations):
        theta = step_gradient_vec(theta, X_b, y, learning_rate)
    return theta

def run(X, y):
    learning_rate = 0.0001
    
    X_b = np.vstack((np.ones(len(X)), X)).T
    theta_0 = np.array([0,0])
    
    num_iterations =100
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(theta_0[0], theta_0[1], fit_error_vec(theta_0, X_b, y))
    print "Running..."
    theta = gradient_descent_runner_vec(X_b, y, theta_0, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, theta[0], theta[1], fit_error_vec(theta, X_b, y))
    return theta