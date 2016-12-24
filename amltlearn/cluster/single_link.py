"""
.. module:: single_link

Single Link
*************

:Description: Single Link agglomerative hierarchical clustering algorithm based
on distances. The distane between any two clusters, is the minimum of distance
between all pairs in these two clusters.


:Authors: kmahyou


:Version: 0.0.1

:Created on: 16/12/2016 15:39 

"""

__author__ = 'kmahyou'

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array
import numpy as np

class SingleLink(BaseEstimator, ClusterMixin, TransformerMixin):
    """Single Link hierarchical clustering
    
    Agglomerative Hierarchical Clustering Method

    Parameters
    ----------
    h : float
        Cut-off distance. Distance-h stopping condition: Distance between any
        pair of clusters in the clustering should be grater than h.
    
    Attributes
    ----------
    clusters_ : array-like, shape=(n_clusters, n_class_inst)
        Every element of this array has a list of instances indices that belong
        to the same cluster

    labels_ : array, shape=(n_instances)
        Label of each instance

    n_clusters : int
        Total number of clusters
    
    height_ : array, shape=(n_instances)
        Indicates when one object joined the cluster of another object

    parent_ : array, shape=(n_instances)
        Indicates who leads the parent cluster. It is useful to track merges

    Examples
    --------
    
    >>> from single_link import SingleLink
    >>> import numpy as np
    >>> X = [[9.21,-0.15],[8.24,1.29],[8.10,-0.45],[8.72,-4.40],
        ... [1.95,21.98],[-1.49,20.50],[-0.76,21.30],[0.95,19.87]]
    >>> sl = SingleLink(h = 3.6).fit(X)
    >>> sl.labels_
    array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int32)
    >>> sl.clusters_
    [[0, 2, 1, 3], [4, 5, 6, 7]]

    See also
    --------
    .. [1] Patra, Bidyut Kr and Nandi, Sukumar and Viswanath, P., "A distance
    based clustering method for arbitrary shaped clusters in large datasets",
    Pattern Recognition, vol. 44, pp. 2862-2870, no.12, 2011.

    """
    
    # INFINITY definition
    INF = np.inf

    def __init__(self, h):
        self.h = h
        self.clusters_ = None
        self.n_clusters = None
        self.labels_ = None
        self.height_ = None
        self.parent_ = None

    def fit(self, X):
        """Computes single link hierarchical clustering

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Dataset set to cluster

        returns
        -------
        self

        """

        X = check_array(X)

        # compute
        clusters, self.height_, self.parent_ = self._fit_process(X)

        self.n_clusters_ = len(clusters.keys())

        # convert from a dictionary to an array, where each element of this
        # array is a list of identifiers that identify an instance
        self.clusters_ = []
        for key in clusters.keys():
            self.clusters_.append(clusters[key])

        # for every instance in X assign it the predicted label
        self.labels_ = np.zeros(X.shape[0], dtype=np.int32)
        for l, ej in enumerate(self.clusters_):
            self.labels_[ej] = l

        return self

    def _fit_process(self, X):
        """Compute the hierarchical clustering"""
     
        # dataset size -> number of samples
        N = X.shape[0]
        
        # indicates when one object joined the cluster of another object 
        height = [self.INF for x in xrange(N)]
    
        # parent node, to track merges -> indicates who leads the parent cluster
        parent = [x for x in xrange(N)]

        # active clusters, when not trivial -> objects not here are still singular
        clusters = dict()
        
        # compute the inter-cluster distance matrix
        dst = self._compute_distances(X)

        # number current iterations. If n_iter == N - 1 stop no mather other
        # criterion
        n_iter = 1
        while True:
            # find two closest clusters
            min_dst, min_x, min_y = self._find_closest_clusters(dst, height, N)

            # check for stopping criterion
            if min_dst > self.h:
                break

            # form a new cluster by merging the two closest newly found clusters
            # merge x in y since y < x. Thus preffer to keep y and drop x
            height[min_x] = min_dst
            parent[min_x] = min_y

            if min_y not in clusters.keys():
                clusters[min_y] = [min_y]

            if min_x not in clusters.keys():
                clusters[min_y].append(min_x)
            else:
                clusters[min_y].extend(clusters[min_x])
                clusters.pop(min_x)

            # update distance matrix for y
            for k in xrange(N):
                dst[k][min_y] = min(dst[k][min_x], dst[k][min_y])
                dst[min_y][k] = min(dst[min_x][k], dst[min_y][k])

            # check for stopping criterion
            if iter == (N - 1):
                break

            n_iter += 1
        
        # if there are some singleton clusters, add them to final clustering
        for x in xrange(N):
            if height[x] < self.INF:
                continue

            if x not in clusters.keys():
                clusters[x] = [x]

        return clusters, height, parent

    def _compute_distances(self, X):
        """Computes the Euclidean distance

        The computed distance is between every pair of instanecs in X. It
        assumes symmetric distances

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Dataset to calculate a pairwise distance

        Returns
        -------
        dst : array, shape=(n_samples, n_samples)
            Computed pair distances amoung elements of X

        """

        N = X.shape[0]
        dst = np.zeros((N, N), dtype=np.double)

        # compute distances
        for i in xrange(N):
            # assumming symmetric distances
            for j in xrange(i):
                dst[i][j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                dst[j][i] = dst[i][j]

        return dst

    def _find_closest_clusters(self, dst, height, N):
        """Find the nearest two clusters in the set of clusters
        
        Parameters
        ----------
        dst : array-like, shape=(n_samples, n_samples)
            Distance matrix

        height : array, shape=(n_samples)
            Indicates when a cluster joined anoother one. It used here to find
            the unjoined clusters

        N : int
            Number of samples

        Returns
        -------
        min_dst : double
            Minimum found distance between two clusters

        min_x : int
            Identifier of the first cluster

        min_y : int
            Identifier of the second cluster

        """
        
        min_dst = self.INF
        min_x, min_y = (-1, -1)
    
        for x in xrange(N):
            if height[x] < self.INF:
                continue
        
            for y in xrange(x):
                if height[y] < self.INF:
                    continue
            
                if dst[x][y] < min_dst:
                    min_dst = dst[x][y]
                    min_x, min_y = x, y

        return min_dst, min_x, min_y

