from collections import defaultdict
import numpy as np

# Python implementation of 
class DBSCAN:
    
    def __init__(self, epsilon = 0.3, min_samples = 10):
        self.eps = epsilon
        self.min_pts = min_samples
        
        # store visited point and clusters
        self.visited_pts = set()
        self.clusters = defaultdict(lambda: defaultdict(set))
        
        
    def fit(self, X):
        self.X = X
        c = -1 # cluster counter
        
        # loop over dataset
        for i in xrange(X.shape[0]):
            
            # skip already visited points
            if i not in self.visited_pts:
                self.visited_pts.add(i)
                neighbor_pts = self.regionQuery(X[i]) # list of point indices
                
                # check if point is in an high density region
                if len(neighbor_pts) >= self.min_pts:
                    # cluster number
                    c += 1
                    # generate new cluster
                    self.expandCluster(i, neighbor_pts, c)
        
        # extract clusters labels from a dictionaries to an array
        labels = self.dict2array()
        return labels
        
    def dict2array(self):
        # initials every point as an outlier (label = -1)
        labels = -1 * np.ones(self.X.shape[0] , dtype=int)

        # Assign labels to points
        for k, v in self.clusters.iteritems():
            labels[list(v['pts_idx'])] = k
        return labels


    def expandCluster(self, i, neighbor_pts, c):
        # initialize new cluster c
        self.clusters[c]["pts_idx"].add(i)

        # Loop while all neighbor point have been visited
        while len(neighbor_pts) > 0:
            # retrieve point on top of the stack
            point = neighbor_pts[0]
            
            # skip already visited points
            if point not in self.visited_pts:
                self.visited_pts.add(point)
                new_neighbor_pts = self.regionQuery(self.X[point]) # list of point indices

                # check if point is in an high density region
                if len(new_neighbor_pts) >= self.min_pts:
                    
                    # add new neighbor point to stack bottom
                    for new_point in new_neighbor_pts:
                        if new_point not in neighbor_pts:
                            neighbor_pts.append(new_point)

            # store all points from every cluster
            pts = set()
            for cluster in self.clusters.itervalues():
                pts.update(cluster.values()[0])
            
            #if point isn't in any cluster, add it to cluster c
            if point not in pts:
                self.clusters[c]["pts_idx"].add(point)
            
            # remove top point from stack    
            neighbor_pts = neighbor_pts[1:]

    def regionQuery(self, x):
        # return a list of point indices that are located within the region of x, bounded by espilon
        return filter(lambda i: np.linalg.norm(self.X[i] - x) <= self.eps, xrange(self.X.shape[0]))
    
