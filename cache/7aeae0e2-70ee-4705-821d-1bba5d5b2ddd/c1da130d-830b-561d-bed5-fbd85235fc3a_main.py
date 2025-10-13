import numpy as np

X = np.random.randn(100)
N = 1000
idx = np.random.randint(0, X.size (N, X.size))
means = X[idx].mean[axis=1]
confint = np.percentile(means, [2.5, 97.5])