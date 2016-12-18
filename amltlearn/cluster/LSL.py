"""
.. module:: LeaderSingleLink

LeaderSingleLink
*****************

:Description: Leader Single Link method. l-SL clustering method is a hybrid
scheme with a combination of Leaders algorithm and Single Link agglomerative
hierarchical clustering techniques. 

The set of of leaders produced by the Leaders clustering method is used as a
representative set of data. Next, the SingleLink method is applied to the
representative set to obtain final clustering.


:Authors: kmahyou


:Version: 0.0.1

:Created on: 17/12/2016 14:11 

"""

__author__ = 'kmahyou'

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import numpy as np

from SingleLink import SingleLink
from Leader import Leader

class LeaderSingleLink(BaseEstimator, ClusterMixin, TransformerMixin):
    """Leader Single Link hierarchical clustering
    
    Hybrid method based in Leaders clsutering and Single Link agglomerative
    hierarchical clustering

    Parameters
    ----------
    h: float
        cut-off distance. Distance-h stopping condition: Distance between any
        pair of clusters in the clustering should be grater than h.
    
    Attributes
    ----------
    clusters_: array, shape=(n_clusters, n_class_inst)
        Every element of this array has a list of instances indices that belong
        to the same cluster
        
    labels_: array, shape=(n_instancecs)
        Label of each instance

    n_clusters: int
        Total number of clusters

    Examples
    --------
    
    >>> from LSL import LeaderSingleLink
    >>> import numpy as np
    >>> np.random.seed(4711)
    >>> a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]],size=[10,])
    >>> b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]],
            size=[10,])
    >>> X = np.concatenate((a, b),)
    >>> lsl = LeaderSingleLink(h=3.6).fit(X)
    >>> lsl.clusters_
    [[0, 1, 3, 6, 8, 5, 2, 9, 4, 7], [10, 13, 14, 17, 18, 11, 12, 15, 19, 16]]
    >>> lsl.labels_
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            dtype=int32)

    See also
    --------
    .. [1] Patra, Bidyut Kr and Nandi, Sukumar and Viswanath, P., "A distance
    based clustering method for arbitrary shaped clusters in large datasets",
    Pattern Recognition, vol. 44, pp. 2862-2870, no.12, 2011.

    """

    def __init__(self, h):
        self.h = h
        self.clusters_ = None
        self.n_clusters = None
        self.labels_ = None

    def fit(self, X):
        """Computes leaders single link hybrid scheme

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape=(n_samples, n_features)
            Dataset set to cluster

        returns
        -------
        self

        """

        # apply leaders method with tau = h / 2
        ld = Leader(self.h / float(2.)).fit(X)

        # apply SL method to L 
        sl = SingleLink(self.h).fit(ld.cluster_centers_)
        
        self.n_clusters_ = sl.n_clusters_
        
        # replace each leader by its followers
        self.clusters_, self.labels_ = self._replace_leaders(sl.clusters_, ld.labels_, X.shape[0])

        return self
 
    def _replace_leaders(self, sl_clusters, ld_labels, N):
        """Replace the leaders in SL clustering by their followers in Leader
        method"""

        clusters = []
        labels = np.zeros(N, dtype=np.int32)
        NR = np.arange(N)
        
        for l, ej in enumerate(sl_clusters):
            tmp = []
            for e in ej:
                tmp.extend(NR[ld_labels == e])

            clusters.append(tmp)
            labels[tmp] = l

        return clusters, labels

if __name__ == "__main__":
    # generate dataset
    np.random.seed(4711)
    a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]],size=[10,])  
    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[10,])

    X = np.concatenate((a, b),)
    
    # run LSL algorithm
    lsl = LeaderSingleLink(h = 3.6)
    lsl.fit(X)

    print "---Clusters---"
    for idx, c in enumerate(lsl.clusters_):
        print "C %d:" %(idx), c

    print "---Labels---"
    print lsl.labels_

