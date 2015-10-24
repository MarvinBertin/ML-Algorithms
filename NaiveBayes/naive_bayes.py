import numpy as np
from collections import defaultdict, Counter
from operator import itemgetter

class NaiveBayes(object):

    def __init__(self, alpha=1):
        self.prior = {}
        self.per_feature_per_label = {}
        self.feature_sum_per_label = {}
        self.likelihood = {}
        self.posterior = {}
        self.alpha = alpha
        self.p = None

    def compute_prior(self, y):
        label_freq = Counter(y)
        num_docs = float(len(y))
        # probability of observing the a particular label
        self.prior = {label: freq/num_docs for label, freq in label_freq.iteritems()}

    def compute_likelihood(self, X, y):
        self.per_feature_per_label = defaultdict(lambda: np.zeros(self.p))
        self.likelihood = defaultdict(lambda: np.zeros(self.p))
        
        # word freq distribution per label
        for label, Xrow in zip(y, X):
            self.per_feature_per_label[label] += Xrow

        # total number of words per label
        self.feature_sum_per_label = {label:sum(Xrow)
                                      for label, Xrow in self.per_feature_per_label.iteritems()}

        # likelihood/probability of observing each word given the label
        for label, Xrow in self.per_feature_per_label.iteritems():
            self.likelihood[label] += (Xrow + self.alpha * np.ones(self.p)) / (self.feature_sum_per_label[label] + self.alpha * self.p)

    def fit(self, X, y):
        self.p = X.shape[1]
        self.compute_prior(y)
        self.compute_likelihood(X, y)

    def predict(self, X):
        self.posterior = defaultdict(float)
        y_hat = []
        for Xrow in X:
            for label in self.prior.iterkeys():
                self.posterior[label] = Xrow.dot(np.log(self.likelihood[label])) + np.log(self.prior[label])
            y_hat.append(sorted(self.posterior.items(), key=itemgetter(1), reverse=True)[0][0])

        return y_hat

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

