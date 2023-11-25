# Import necessary libraries
from scipy.linalg import svdvals, schur
from scipy.stats import zscore
import numpy as np
import pandas as pd

# Define functions for calculating control metrics

def Normalization(A):
    """
    Normalize the adjacency matrix A.

    :param A: Adjacency matrix.
    :return: Normalized adjacency matrix.
    """
    return A / (1 + max(svdvals(A)))

def PyC_AverageControl(A):
    """
    Calculate Average Controllability for each node.

    Average Controllability measures the ease by which input at a node can steer the system into many easily-reachable states.

    :param A: Normalized adjacency matrix.
    :return: Vector of average controllability values for each node.
    """
    A = Normalization(A)
    T, U = schur(A, output='real')  # Schur decomposition for stability
    midMat = np.square(U.T)
    v = np.diag(T)
    P = np.tile(1 - np.square(v), (A.shape[0], 1)).T
    values = np.sum(midMat / P, axis=0).T
    return values

def PyC_ModalControl(A):
    """
    Calculate Modal Controllability for each node.

    Modal Controllability indicates the ability of a node to steer the system into difficult-to-reach states.

    :param A: Normalized adjacency matrix.
    :return: Vector of modal controllability values for each node.
    """
    A = Normalization(A)
    T, U = schur(A, output='real')  # Schur decomposition for stability
    eigVals = np.diag(T)
    N = A.shape[0]
    phi = np.zeros(N)
    for i in range(N):
        phi[i] = np.sum(U[i, :]**2 * (1 - eigVals**2))
    return phi

# Process data frame for control metrics

# Assuming df is your DataFrame and it contains a column 'A_matrice' with adjacency matrices
df['A_Norm'] = df['A_matrice'].apply(Normalization)

# Calculating control metrics
df['Average'] = df['A_matrice'].apply(PyC_AverageControl)
df['Modal'] = df['A_matrice'].apply(PyC_ModalControl)
df['TimeConstant'] = df['A_Norm'].apply(np.diag)
