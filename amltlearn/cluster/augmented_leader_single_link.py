"""
.. module:: augmented_leader_single_link

Augmented Leader Single Link
****************************

:Description: Augmented Leader Single Link method. al-SL clustering method is a
hybrid scheme with a combination of Leaders algorithm ans Single Link
agglomerative hierarchical clustering techniques. Although it is similar to
LeaderSingleLink algorithm, al-SL perform a merging between any two clusters
resulting from the previous steps if the distance between them i less or equal
than the cut-off, h.

The set of of leaders produced by the Leaders clustering method is used as a
representative set of data. Next, the SingleLink method is applied to the
representative set to obtain final clustering. After, it merges any two
clusters if the distance is less or equal to h.


:Authors: kmahyou


:Version: 0.0.1

:Created on: 17/12/2016 17:28 

"""

__author__ = 'kmahyou'

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array
import numpy as np

from single_link import SingleLink
from Leader import Leader

class AugmentedLeaderSingleLink(BaseEstimator, ClusterMixin, TransformerMixin):
    """Augmented Leader Single Link hierarchical clustering
    
    Hybrid method based in Leaders clsutering and Single Link agglomerative
    hierarchical clustering

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
    
    Examples
    --------
    
    >>> from augmented_leader_single_link import AugmentedLeaderSingleLink
    >>> import numpy as np
    >>> X = [[9.21,-0.15],[8.24,1.29],[8.10,-0.45],[8.72,-4.40],
        ... [1.95,21.98],[-1.49,20.50],[-0.76,21.30],[0.95,19.87]]
    >>> alsl = AugmentedLeaderSingleLink(h = 3.6).fit(X)
    >>> alsl.labels_
    array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int32)
    >>> alsl.clusters_
    [[0, 2, 1, 3], [4, 5, 6, 7]]

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
        """Computes augmented leaders single link hybrid scheme

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Dataset set to cluster

        returns
        -------
        self

        """

        X = check_array(X)

        # apply leaders method with tau = h / 2
        ld = Leader(self.h / float(2.)).fit(X)

        # apply SL method to L 
        sl = SingleLink(self.h).fit(ld.cluster_centers_)
        
        # merging part -> merge closest clusters
        clusters_ = self._merge_clusters(X, sl.clusters_, ld.labels_)

        self.n_clusters_ = len(clusters_)
        
        self.clusters_, self.labels_ = self._replace_leaders(clusters_, ld.labels_, X.shape[0])

        return self

    def _merge_clusters(self, X, sl_clusters, ld_labels):
        """Merge the resulting clusters that are closer together"""

        k = len(sl_clusters)
        cutoff = 2 * self.h
        NR = np.arange(X.shape[0])

        # find potential leaders to merge
        S = set()
        for bi in xrange(len(sl_clusters)):
            for bj in xrange(bi + 1, len(sl_clusters)):
            # for every pair of clusters do
                dst, li, lj = self._distance_cluster(X, 
                        sl_clusters[bi], sl_clusters[bj])

                if dst > cutoff:
                    continue
                
                # find potential leaders for possible merging
                Lbi, Lbj = set(), set()
                
                for lx in sl_clusters[bi]:
                    if self._distance(lx, lj) <= cutoff: Lbi.add(lx)

                for ly in sl_clusters[bj]:
                    if self._distance(ly, li) <= cutoff: Lbj.add(ly)

                S = S.union(Lbi)
                S = S.union(Lbj)

        # merge potential leaders if the distance is lower or equal than h
        if len(S) != 0:
            # apply leaders method with tau = h / 2
            ld = Leader(self.h / float(2.)).fit(X)

            S = list(S)
            followers = self._compute_followers(S, ld.labels_, X.shape[0])

            # for every pair of clusters do
            for bi in xrange(len(sl_clusters)):
                for bj in xrange(bi + 1, len(sl_clusters)):
                    dst, _, _ = self._distance_cluster(X, 
                                    sl_clusters[bi], sl_clusters[bj])
                    
                    if dst > cutoff:
                        continue

                    isbreak = False
                    for la in S:
                        for lb in S:
                            if not(la in sl_clusters[bi] and lb in
                                    sl_clusters[bj]):
                                continue

                            # find two nearest followers of lb and lb
                            x, y = -1, -1
                            dst_xy = np.inf
                            for x_a in followers[la]:
                                for y_b in followers[lb]:
                                    tmp = self._distance(X[x_a], X[y_b])
                                    if tmp < dst_xy:
                                        dst_xy = tmp
                                        x = x_a
                                        y = y_b
                            
                            # if the distance is less or equal to h merge both
                            # clusters
                            if self._distance(X[x], X[y]) <= self.h:
                                sl_clusters[bi].extend(sl_clusters[bj])
                                isbreak = True
                                break

                        if isbreak:
                            break

            # perform the merging
            rm = []
            for x in xrange(len(sl_clusters)):
                for y in xrange(x + 1, len(sl_clusters)):
                    if y in rm:
                        continue

                    l = [k for k in sl_clusters[x] if k in sl_clusters[y]]

                    if len(l) != 0:
                        sl_clusters[x] = list(set(sl_clusters[x] +
                            sl_clusters[y]))
                        rm.append(y)
                   
            sl_clusters = [sl_clusters[x] for x in xrange(len(sl_clusters)) if x not in
                    rm]

        return sl_clusters
 
    def _distance_cluster(self, X, C1, C2,):
        """Calculates the Euclidean distance between a pair of clusters

        distance(C1, C2) = min{||xi - xj|| | xi in C1, xj in C2}

        """

        li, lj = -1, -1
        min_dst = np.inf

        for i in C1:
            for j in C2:
                dst = self._distance(X[i], X[j])

                if dst < min_dst:
                    min_dst = dst
                    li = i
                    lj = j

        return min_dst, li, lj

    def _distance(self, x, y):
        """Calculates the Euclidean distance between two given points"""

        return np.sqrt(np.sum((x - y) ** 2))

    def _compute_followers(self, leaders, ld_clusters, N):
        """Compute the followers for every leader"""

        NR = np.arange(N)
        follow = dict()

        for l in leaders:
            follow[l] = NR[ld_clusters == l]

        return follow

    def _replace_leaders(self, sl_clusters, ld_clusters, N):
        """Replace the leaders in SL clustering by their followers in Leader
        method"""

        clusters = []
        labels = np.zeros(N, dtype=np.int32)
        NR = np.arange(N)
        
        for l, ej in enumerate(sl_clusters):
            tmp = []
            for e in ej:
                tmp.extend(NR[ld_clusters == e])

            clusters.append(tmp)
            labels[tmp] = l

        return clusters, labels

