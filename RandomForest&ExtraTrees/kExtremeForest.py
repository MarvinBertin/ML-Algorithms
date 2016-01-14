from DecisionTree import DecisionTree
import numpy as np
from collections import Counter
from RandomForest import RandomForest
from kExtremeTree import kExtremeTree

class kExtremeForest(RandomForest):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features, impurity_criterion, prune = False):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        RandomForest.__init__(self, num_trees, num_features, impurity_criterion)
        self.prune = prune

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Return a list of num_trees DecisionTrees.
        '''
        forest = []
        for i in xrange(num_trees):
            sample_indices = np.random.choice(X.shape[0], num_samples, \
                                              replace=True)
            sample_X = np.array(X[sample_indices])
            sample_y = np.array(y[sample_indices])
            
            dt = kExtremeTree(impurity_criterion = self.impurity_criterion,
                              num_features = self.num_features,
                              prune = self.prune)
            dt.fit(sample_X, sample_y)
            forest.append(dt)
        return forest
