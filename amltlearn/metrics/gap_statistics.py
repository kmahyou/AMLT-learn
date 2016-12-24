"""
.. module:: metrics

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
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.cluster import KMeans
import numpy as np

def gap_statistics(X, K = 10, B = 10, ref_method = "unif", 
        random_state = None, **kwargs):
    """Gap statistics to estimate the number of clusters in a dataset

    Parameters
    ----------
    X : array-like, shape(n_samples, n_features)
        Dataset to estimate the clusters in

    K : int, default=10
        Number of clusters to consider

    B : int, default=10
        Number of Monte Carlo samples ("bootstrap")

    ref_method : {"unif", "pc"}, default="unif"
        Which method to use to generate reference data sets.
        
        "unif": generate each reference feature uniformly over the range of
        the obversed values for that particular feature.
        
        "pc": generate the reference features from a uniform distribution over a
        box aligned with the principal components of the data using SVD
        decomposition.

        "unif" has the advantage of simplicity. Whereas "pc" takes into
        account the shape of the data distribution.

    random_state : int or numpy.RandomState, default=None
        The generator used to generate the reference samples. If an integer
        given, it fixes the seed. Defaults to the global numpy random
        generator.

    **kwargs:
        Additional parameters for KMeans algorithm

    Returns
    -------
    k_hat : int
        Estimated number of clusters

    W_obs : array, shape(K)
        Within-cluster dispersion for different K's for given dataset X

    W_exp_mean : array, shape(K)
        Within-cluster dispersion for different K's for generated dataset. This
        is a mean of B generated datasets

    W_exp_std : array, shape(K)
        Standard deviation of dispersion for different K's for generated
        dataset.

    Examples
    --------
    
    >>> from amltlearn.metrics import gap_statistics
    >>> import numpy as np
    >>> X = [[9.21188389,-0.15188589], [8.88937431,-0.33937464],
        ... [10.76840064,2.95244645], [-1.41259576,21.66814229],
        ... [1.48256428,19.71376506], [1.49912084,19.44886828]]
    >>> kmeans_args = {'n_init': 10,'max_iter': 500}
    >>> k, W_obs, W_exp, W_exp_std = gap_statistics(X, K = 4, B = 10, 
        ... random_state=42, **kmeans_args)
    >>> k
    2
    >>> W_obs
    array([ 2.85704081,  0.91999294,  0.28506657, -1.83498871])
    >>> W_exp
    array([ 2.48145812,  1.75309355,  1.17625568,  0.43859918])
    >>> W_exp_std
    array([ 0.21222606,  0.23449418,  0.24864987,  0.40315318])
    
    See also
    --------
    .. [1] Tibshirani, Robert and Walther, Guenther and Hastie, Trevor,
    "Estimating the number of clusters in a data set via the gap statistics",
    Journal of the Royal Statistical Society B, vol. 63, pp. 411-423, 2001.
    
    """

    X = check_array(X)

    random_state = check_random_state(random_state)

    # check if the reference generator method is the required one
    if ref_method == "unif":
        gen_ref = generate_uniform_points
    elif ref_method == "pc":
        gen_ref = generate_uniform_points_svd
    else:
        raise ValueError("Reference method must be 'unif' or 'pc', got"
                " %s" % str(ref_method))

    # cluster the observed data with different k=1..K and calculate W_obs
    W_obs = np.zeros(K)
    for k in xrange(K):
        # cluster the data
        kmeans = KMeans(n_clusters = k + 1, **kwargs).fit(X)
    
        # dispersion
        W_obs[k] = dispersion(X, kmeans.labels_)

    # generate B reference datasets, cluster it with k=1..K and calculate W_exp
    W_exp = np.zeros((B, K))
    for b in xrange(B):
        # generate reference dataset
        #U = generate_uniform_points_svd(X)
        U = gen_ref(X, random_state)

        for k in xrange(K):
            # cluster the data
            kmeans = KMeans(n_clusters = k + 1, **kwargs).fit(U)

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

def generate_uniform_points(X, random_state):
    """Generate Uniform points in the observed data X range

    Generate each reference feature uniformly over the range of the observed
    values for that particular feature

    Parameters
    ---------
    X : array-like, shape=(n_samples, n_features)
        Dataset to generate features range from

    random_state : numpy.RandomState
        The generator used to generate the reference samples.
    
    Returns
    -------
    U : array-like, shape=(n_samples, n_features)
        uniformly generated dataset, where each uniform feature is generated in
        the range of each observed feature

    """

    # find the min/max values for each feature
    mins = np.min(X, axis = 0)
    maxs = np.max(X, axis = 0)

    # number of samples
    N = X.shape[0]

    # for each feature generate N uniform points in range (mins, maxs]
    U = [random_state.uniform(low=mins[f], high=maxs[f], size=N) for f in
            xrange(len(mins))]

    return np.asarray(U).T

def generate_uniform_points_svd(X, random_state):
    """Generate Uniform points in the observed data X range

    Generate the reference features from a uniform distribution over a box
    aligned with the principal components of the data. Using singular value
    decomposition (UDV)

    Parameters
    ---------
    X : array-like, shape=(n_samples, n_features)
        Dataset to generate features range from

    random_state : numpy.RandomState
        The generator used to generate the reference samples.
    
    Returns
    -------
    U : array-like, shape=(n_samples, n_features)
        Uniformly generated dataset, where each uniform feature is generated in
        the range of each observed feature

    """

    # calculate de decomposition of the data
    U, s, V = np.linalg.svd(X, full_matrices=True)

    # tranform it and draw uniform points in the range of columns
    X_a = np.dot(X, V)    
    Z = generate_uniform_points(X_a, random_state)

    # back-tranform and return the result
    return np.dot(Z, np.transpose(V))

def dispersion(X, labels):
    """Calculate the within-cluster dispersion
    
    The dispersion of one cluster (C) is defined as follows:
    dispersion(r) = sum((1/(2*Nr)) * Dr) where Nr = |Cr| and Dr is the sum of
    pairwise distances for all points in cluster r.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Dataset to estimate the clusters in

    labels : array, shape=(n_samples)
        Label of each instance in X

    Returns
    -------
    dispersion: dispersion
        within-cluster sum dispersion for a particular K, where K =
        len(unique(labels))
    
    """

    # compute the within-cluster dispersion
    return sum([sum(pdist(X[labels == c])) / (2. * len(labels[labels == c])) \
                for c in np.unique(labels)])

