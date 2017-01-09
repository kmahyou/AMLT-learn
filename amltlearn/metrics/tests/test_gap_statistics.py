"""Testing for gap statistics"""

import numpy as np

from amltlearn.metrics import gap_statistics

if __name__ == "__main__":
    # generate dataset
    np.random.seed(4711)
    a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]],size=[10,])  
    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[10,])

    X = np.concatenate((a, b))

    # run gap statistics algorithm
    kmeans_args = {'n_init': 10, 'max_iter': 500}
    k, W_obs, W_exp, W_exp_std, gaps = gap_statistics(X, ref_method="unif",
            random_state=42, **kmeans_args)

    print "---W obs---"
    print W_obs
    print "---W exp---"
    print W_exp
    print "---W exp std---"
    print W_exp_std
    print "---Number of Estimated Clusters---"
    print "k --> %d" %(k)
    print "---Gaps---"
    print gaps




