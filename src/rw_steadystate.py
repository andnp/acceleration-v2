import numpy as np

N = 5

pl = 0.5
pr = 0.5

P = np.zeros((N, N))
P[0, 1] = pr
P[0, N // 2] = pl
P[N - 1, N - 2] = pl
P[N - 1, N // 2] = pr
for i in range(1, N - 1):
    P[i, i - 1] = pl
    P[i, i + 1] = pr

def matrix_power(X, n):
    A = X
    for _ in range(n):
        A = np.dot(A, X)

    return A

A = matrix_power(P, 100000)
print(np.mean(A, axis=0))
