
import pandas as pd
from scipy.linalg import svdvals, schur

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

def PyC_ControlMetrics(df, adjacency_col):
    """
    Calculate control metrics for a DataFrame containing adjacency matrices.

    :param df: DataFrame containing the data.
    :param adjacency_col: Column name containing adjacency matrices.
    :return: DataFrame with additional columns for control metrics.
    """
    # Normalization
    df['A_Norm'] = df[adjacency_col].apply(Normalization)

    # Calculating control metrics
    df['Average'] = df[adjacency_col].apply(PyC_AverageControl)
    df['Modal'] = df[adjacency_col].apply(PyC_ModalControl)
    df['TimeConstant'] = df['A_Norm'].apply(lambda x: np.diag(x))

    return df

# Example usage:
# df = pd.read_csv('your_data.csv') # Load your data
# df = PyC_ControlMetrics(df, 'A_matrices')
# print(df.head())
