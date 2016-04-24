import random
import numpy as np
# Source: Jared Thompson

class LDA:
    
    def __init__(self, alpha=0.01):
        
        self.alpha = alpha # alpha is the gibbs smoothing parameter
        self.document_topic_counts = None # a list of Counters, one for each document
        self.topic_word_counts = None # a list of Counters, one for each topic
        self.topic_counts = None # a list of numbers, one for each topic
        self.document_lengths = None # a list of numbers, one for each document
        self.document_topics = None # the documents are assigned a topic that they belong to
        self.corpus = set(word for document in documents for word in document) # The number of distinct words
        
        self.N = None # this is the size of the corpus
        self.D = None
     
    def _draw_sample(self, weights):
        
        """returns i with probability weights[i] / sum(weights)"""
        total = sum(weights)
        rnd = total * random.random() # uniform between 0 and total
        # This code is a little meh. Could be improved
        for i, w in enumerate(weights):
            rnd -= w # this little filter checks if the random number fell below
            # the current weight value
            if rnd <= 0:
                return i # if so, return the bin number of the weight

    def _Z_dist(self,topic, d, alpha=0.1):
        # This is the current value of Zt,d for a given topic and doc (we do not explicitly store Z)
        
        '''Estimated probability of p(topic|document, alpha)'''
        return ((self.document_topic_counts[d][topic] + alpha) /
                (self.document_lengths[d] + self.K * alpha))

    def _W_dist(self, word, topic, beta=0.1):
        # This is the current value of Wn,t for a given topic and doc (we do not explicitly store W)
        
        '''Estimated probability of p(word|topic, beta)'''
        return ((self.topic_word_counts[topic][word] + beta) /
                (self.topic_counts[topic] + self.N * beta))
            
    def _topic_weight(self,d, word, k):
        '''given a document and a word in that document,
        return the weight for the kth topic'''
        # Thus the computation is P(word|topic)P(topic|document)
        return self._W_dist(word, k) * self._Z_dist(k, d)

    def _choose_new_topic(self, d, word):
    
        # construct the current weights P(word|document) 
        # of all K topics for that document - Based on current histograms (t-1)!!
        document_weights = [self._topic_weight(d, word, k)
                            for k in range(self.K)]

        return self._draw_sample(document_weights) 
        # takes a random sample from among all P(word|document, topic)
    
    def _gibbs_sample(self):
        
        for d in range(self.D): # D index actually, documents[d] is our set of docs...
            for i, (word, topic) in enumerate(zip(self.X[d],
                                              self.document_topics[d])): # study this loop carefully
            
                # we are really going over the distribution of document words and topics Wn,d  
                
                # downdate
                # we remove this word / topic from the counts
                # so that it doesn't influence the weights
                self.document_topic_counts[d][topic] -= 1 # Zn,t
                self.topic_word_counts[topic][word] -= 1 # W_t,n
                self.topic_counts[topic] -= 1 # T[t] This refers to Beta_k
                self.document_lengths[d] -= 1 # N[d] length of each doc
                
                # choose a new topic randomly for this word based on the current weights within the doc
                new_topic = self._choose_new_topic(d, word)
                # take this new assignment and assign it to the document
                self.document_topics[d][i] = new_topic 
                
                
                # update
                # and now add the newly chosen topic back to the counts
                self.document_topic_counts[d][new_topic] += 1
                self.topic_word_counts[new_topic][word] += 1
                self.topic_counts[new_topic] += 1
                self.document_lengths[d] += 1                

    def _extract_word_counts(self):
        print self.document_topics
        for d in range(self.X.shape[0]):
            for word, topic in zip(self.X[d], self.document_topics[d]):
                self.document_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][word] += 1
                self.topic_counts[topic] += 1
        
        print self.document_topic_counts
    
    def _assign_storage(self):
        # count the number of documents and get the corpus
        self.document_topic_counts = [Counter() for _ in range(self.X.shape[0])] 
        self.topic_word_counts = [Counter() for _ in range(self.K)] 
        self.topic_counts = [0 for _ in range(self.K)] 
        self.document_lengths = [len(document) for document in self.X]
        self.corpus = set(word for document in self.X for word in document)

        # randomly initialize the document topics 
        # to a random uniform distribution - this is a place for improvement
        self.document_topics = [[random.randrange(self.K) for word in document] for document in self.X]
        
           
                
    def fit(self, X, K=1, iterations=100):
        
        #In general, if we don't know K beforehand, we have to have a way to find it.
        # Usally DPGMM is used 
        
        self.X = np.asarray(X)
        self.D = self.X.shape[0]
        self.K = K
        
        # assign memory to storage structures
        self._assign_storage()
        self.N = len(self.corpus) # this is the size of the corpus
        
        # extract counts from the document set
        self._extract_word_counts()
        
        # perform gibbs sampling to reconstruct the prior
        for i in xrange(iterations):
            self._gibbs_sample()

    def predict(self):
        
        output = {}
        for k, word_counts in enumerate(self.topic_word_counts):
            for word, count in word_counts.most_common():
                if count > 0:
                    if k in output:
                        output[k].append(word)
                    else:
                        output[k] = []
                        output[k].append(word)
                        
        # From here you would want to define a way of classifying the input.
        # One way of doing so would be to determine the 
        # unique words belonging to a document (row in X) and assign a confidence 
        # of topic assignment to the input document based on
        # a fraction of words in topic. You would want to weight the
        # the word found by the counts of the word as given above
        # and in the document. This gives a measure of relative importance.
        
        
        return output