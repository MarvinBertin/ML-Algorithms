import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        '''
        # Initialize weights to 1 / n_samples
        n = float(x.shape[0])
        sample_weight = np.ones(n) / n
        
        # For each of n_estimators, boost
        for i in xrange(self.n_estimator):
            estimator, sample_weight, estimator_weight = self._boost(x, y, sample_weight)
            
            # Append estimator, sample_weights and error to lists
            self.estimators_.append(estimator)
            self.estimator_weight_[i] = estimator_weight


    def _boost(self, x, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''

        estimator = clone(self.base_estimator)

        # Fit according to sample weights, emphasizing certain data points
        estimator.fit(x, y, sample_weight = sample_weight)

        # Calculate instances incorrectly classified and store as incorrect
        y_pred = estimator.predict(x)
        misclassified = y_pred != y 
        
        # calculate fraction of error as estimator_error
        estimator_error = np.sum(sample_weight * misclassified) / np.sum(sample_weight)
        
        # Update estimator weights
        estimator_weight = self.learning_rate * np.log((1 - estimator_error) / estimator_error)
        
        # Update sample weights
        sample_weight *= np.exp(estimator_weight * misclassified)
        
        return estimator, sample_weight, estimator_weight


    def predict(self, x):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''
        # get predictions from tree family
#         vpred = np.vectorize(*args.predict(x))
#         predictions = vpred(self.estimators)
        predictions = np.array(map(lambda est: est.predict(x), self.estimators_))

        # set negative predictions to -1 instead of 0 (so we have -1 vs. 1)
        predictions[predictions == 0] = -1
        
        return self.estimator_weight_.dot(predictions) >= 0

    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''
        y_pred = self.predict(x)
        return np.mean(y == y_pred)
