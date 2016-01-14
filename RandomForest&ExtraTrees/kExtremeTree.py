import numpy as np
import math
from collections import Counter

from DecisionTree import DecisionTree
from RandomForest import RandomForest

class kExtremeTree(DecisionTree):
    
    def __init__(self, impurity_criterion, num_features = None, prune = False):
        
        DecisionTree.__init__(self, impurity_criterion='entropy')
        self.k = num_features
        self.pruning = prune
    
    def fit(self, X, y, feature_names=None):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - feature_names: numpy array of strings
        OUTPUT: None
        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each
        row a data point.
        y is a 1 dimensional array with each value being the corresponding
        label.
        feature_names is an optional list containing the names of each of the
        features.
        '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        # Create True/False array of whether the variable is categorical
        is_categorical = lambda x: isinstance(x, str) or \
                                   isinstance(x, bool) or \
                                   isinstance(x, unicode)
        self.categorical = np.vectorize(is_categorical)(X[0])
        
        # Check number of features selected does not exceed number of columns
        if self.k > X.shape[1]:
            raise ValueError("Number of features exceed the number of columns.")
        
        # set number of features considered per split
        if not self.k:
            self.k = X.shape[1]

        self.root = self._build_tree(X, y)
        
        # prune tree
        if self.pruning:
            self.prune(X, y, node=self.root)
    
    def _choose_split_index(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)
        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.
        Return None, None, None if there is no split which improves information
        gain.
        Call the method like this:
        >>> index, value, splits = self._choose_split_index(X, y)
        >>> X1, y1, X2, y2 = splits
        '''

        split_index, split_value, splits = None, None, None
        max_gain = 0
        
        # Only select k random features at each split
        feature_idx = np.random.choice(X.shape[1], size = self.k, replace = False)
        
        for i in feature_idx:
            values = np.unique(X[:, i])
            if len(values) < 2:
                continue
            for val in values:
                temp_splits = self._make_split(X, y, i, val)
                X1, y1, X2, y2 = temp_splits
                gain = self._information_gain(y, y1, y2)
                if gain > max_gain:
                    max_gain = gain
                    split_index, split_value = i, val
                    splits = temp_splits
        return split_index, split_value, splits
    
    def prune(self, X, y, node=None):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - node: TreeNode root
        OUTPUT:
            None
        Recursively checks for leaves and compares error rate before and after
        merging the leaves.  If merged improves error rate, merge leaves.
        '''

        if node == None:
            self.root = node

        if node.left.leaf == False:
            self.prune(X, y, node = node.left)

        if node.right.leaf == False:
            self.prune(X, y, node = node.right)

        if node.right.leaf and node.left.leaf:
            
            leaf_y = self.predict(X)
            merged_classes = node.right.classes + node.left.classes
            merged_name = merged_classes.most_common(1)[0][0]
            merged_y = [merged_name] * len(y)

            leaf_score = np.mean(leaf_y == y)
            merged_score = np.mean(merged_y == y)
            
            if merged_score >= leaf_score:
                node.leaf = True
                node.classes = merged_classes
                node.name = merged_name
                node.left = None
                node.right = None