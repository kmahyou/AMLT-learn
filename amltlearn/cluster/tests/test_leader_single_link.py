"""Testing for Single Link"""

import numpy as np

from amltlearn.cluster import LeaderSingleLink

# LSL
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
