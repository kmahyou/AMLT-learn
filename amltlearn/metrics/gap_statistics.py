"""
.. module:: evaluation

Gap Statistics
*************

:Description: Gap Statistics estimate the number of clusters in a data set.

The technique uses the output of any clustering algorithm, comparing the change
in withi-cluster dispersion with that expected under an appropiate reference
null distribution.

:Authors: kmahyou


:Version: 0.0.1

:Created on: 20/12/2016 16:13 

"""

__author__ = 'kmahyou'

from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
import numpy as np

def gap(X, K = 10, B = 10, ref_method="obsfeat"):
    """Gap statistics to estimate the number of clusters in a dataset

    Parameters
    ----------
    X: {array-like, sparse matrix}, shape(n_samples, n_features)
        Dataset to estimate the clusters in

    K: int, default=10
        number of clusters to consider

    B: int, default=10
        number of Monte Carlo samples ("bootstrap")

    ref_method: "obsfeat" or "svd", default="obsfeat"
        which method to use to generate reference data sets.
        
        "obsfeat": generate each reference feature uniformly over the range of
        the obversed values for that particular feature.
        
        "svd": generate the reference features from a uniform distribution over a
        box aligned with the principal components of the data using SVD
        decomposition.

        "obsfeat" has the advantage of simplicity. Whereas "svd" takes into
        account the shape of the data distribution.

    Returns
    -------
    k_hat: int
        estimated number of clusters

    W_obs: array-like, shape(K)
        within-cluster dispersion for different K's for given dataset X

    W_exp_mean: array-like, shape(K)
        within-cluster dispersion for different K's for generated dataset. This
        is a mean of B generated datasets

    W_exp_std: array-like, shape(K)
        standard deviation of dispersion for different K's for generated
        dataset.

    Examples
    --------
    
    >>> from gap_statistics import gap
    >>> import numpy as np
    >>> X = np.asarray([[9.21188389,-0.15188589], [8.88937431,-0.33937464],
        ... [10.76840064,2.95244645], [-1.41259576,21.66814229],
        ... [1.48256428,19.71376506], [1.49912084,19.44886828]])
    >>> k, W_obs, W_exp, W_exp_std = gap(X, K = 4, B = 10)
    >>> k
    2
    >>> W_obs
    array([ 2.85704081,  0.91999294,  0.28506657, -1.83498871])
    >>> W_exp
    array([ 2.52347108,  1.80307407,  1.08936194,  0.37972856])
    >>> W_exp_std
    array([ 0.18898569,  0.17419498,  0.26456884,  0.25577568])
    
    See also
    --------
    .. [1] Tibshirani, Robert and Walther, Guenther and Hastie, Trevor,
    "Estimating the number of clusters in a data set via the gap statistics",
    Journal of the Royal Statistical Society B, vol. 63, pp. 411-423, 2001.
    
    """

    n_init_ = 5
    max_iter_ = 50
    
    # check if the reference generator method is the required one
    if ref_method == "obsfeat":
        gen_ref = generate_uniform_points
    elif ref_method == "svd":
        gen_ref = generate_uniform_points_svd
    else:
        raise ValueError("Reference method must be 'obsfeat' or 'svd', got"
                " %s" % str(ref_method))


    # cluster the observed data with different k=1..K and calculate W_obs
    W_obs = np.zeros(K)
    for k in xrange(K):
        # cluster the data
        kmeans = KMeans(n_clusters = k + 1, n_init = n_init_, 
                max_iter = max_iter_).fit(X)
    
        # dispersion
        W_obs[k] = dispersion(X, kmeans.labels_)

    # generate B reference datasets, cluster it with k=1..K and calculate W_exp
    W_exp = np.zeros((B, K))
    for b in xrange(B):
        # generate reference dataset
        #U = generate_uniform_points_svd(X)
        U = gen_ref(X)

        for k in xrange(K):
            # cluster the data
            kmeans = KMeans(n_clusters = k + 1, n_init = n_init_, 
                    max_iter = max_iter_).fit(U)

            # dispersion
            W_exp[b, k] = dispersion(U, kmeans.labels_)

    # convert to log values
    W_obs = np.log(W_obs)
    W_exp = np.log(W_exp)
    
    # compute the estimated gap statistics
    gaps = (1/float(B)) * np.sum(W_exp - W_obs, axis = 0)

    # compute the mean and standard deviation
    W_exp_mean = np.mean(W_exp, axis = 0)
    W_exp_std = np.std(W_exp, axis = 0) * np.sqrt(1 + (1/float(B)))

    # choose the number of clusters
    k_hat = np.min([k+1 for k in xrange(K-1) \
                    if gaps[k] >= gaps[k+1]-W_exp_std[k+1]])
    
    return k_hat, W_obs, W_exp_mean, W_exp_std

def generate_uniform_points(X):
    """Generate Uniform points in the observed data X range

    Generate each reference feature uniformly over the range of the observed
    values for that particular feature

    Paramters
    ---------
    X: {array-like, sparse matrix}, shape=(n_samples, n_features)
        dataset to generate features range from

    Returns
    -------
    U: array-like, shape=(n_samples, n_features)
        uniformly generated dataset, where each uniform feature is generated in
        the range of each observed feature

    """

    # find the min/max values for each feature
    mins = np.min(X, axis = 0)
    maxs = np.max(X, axis = 0)

    # number of samples
    N = X.shape[0]

    # for each feature generate N uniform points in range (mins, maxs]
    U = [np.random.uniform(low=mins[f], high=maxs[f], size=N) for f in
            xrange(len(mins))]

    return np.asarray(U).T

def generate_uniform_points_svd(X):
    """Generate Uniform points in the observed data X range

    Generate the reference features from a uniform distribution over a box
    aligned with the principal components of the data. Using singular value
    decomposition (UDV)

    Paramters
    ---------
    X: {array-like, sparse matrix}, shape=(n_samples, n_features)
        dataset to generate features range from

    Returns
    -------
    U: array-like, shape=(n_samples, n_features)
        uniformly generated dataset, where each uniform feature is generated in
        the range of each observed feature

    """

    # calculate de decomposition of the data
    U, s, V = np.linalg.svd(X, full_matrices=True)

    # tranform it and draw uniform points in the range of columns
    X_a = np.dot(X, V)    
    Z = generate_uniform_points(X_a)

    # back-tranform and return the result
    return np.dot(Z, np.transpose(V))


def dispersion(X, labels):
    """Calculate the within-cluster dispersion
    
    The dispersion of one cluster (C) is defined as follows:
    dispersion(r) = sum((1/(2*Nr)) * Dr) where Nr = |Cr| and Dr is the sum of
    pairwise distances for all points in cluster r.

    Parameters
    ----------
    X: {array-like, sparse matrix}, shape=(n_samples, n_features)
        dataset to estimate the clusters in

    labels: array, shape=(n_samples)
        label of each instance in X

    Returns
    -------
    dispersion: dispersion
        within-cluster sum dispersion for a particular K, where K =
        len(unique(labels))
    
    """

    # compute the within-cluster dispersion
    return sum([sum(pdist(X[labels == c])) / (2. * len(labels[labels == c])) \
                for c in np.unique(labels)])

if __name__ == "__main__":
    # generate dataset
    
    #np.random.seed(4711)
    #a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]],size=[10,])  
    #b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[10,])

    #X = np.concatenate((a, b))

    X = np.asarray([[9.21188389,-0.15188589],
            [8.88937431,-0.33937464],
            [10.76840064,2.95244645],
            [8.24213268,1.29094802],
            [5.7967009,-5.83776714],
            [6.3499309,0.63959515],
            [8.1057123,-0.45887277],
            [8.72084884,-4.40444487],
            [8.83500513,-0.3916611 ],
            [10.4006121,2.71240817],
            [1.95503402,21.98602715],
            [-1.4985815,20.50349583],
            [-0.7664081,21.30693205],
            [2.27768001,21.62763958],
            [0.95901841,19.87247968],
            [-1.27562588,19.26898089],
            [-0.35050873,23.4198941 ],
            [1.49912084,19.44886828],
            [1.48256428,19.71376506],
            [-1.41259576,21.66814229]])

    # run gap statistics algorithm
    k, W_obs, W_exp, W_exp_std = gap(X, ref_method="obsfeat")

    print "---W obs---"
    print W_obs
    print "---W exp---"
    print W_exp
    print "---W exp std---"
    print W_exp_std
    print "---Number of Estimated Clusters---"
    print "k --> %d" %(k)

