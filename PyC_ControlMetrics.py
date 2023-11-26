
import pandas as pd
import numpy as np
from scipy.linalg import svdvals, schur
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator


def Normalization(A):
    """
    Normalize the adjacency matrix A.

    Parameters:
    A (numpy.ndarray): Adjacency matrix to be normalized.

    Returns:
    numpy.ndarray: Normalized adjacency matrix.
    """
    return A / (1 + max(svdvals(A)))

def PyC_AverageControl(A):
    """
    Calculate Average Controllability for each node in a network.

    Parameters:
    A (numpy.ndarray): Normalized adjacency matrix of the network.

    Returns:
    numpy.ndarray: A vector of average controllability values for each node.
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
    Calculate Modal Controllability for each node in a network.

    Parameters:
    A (numpy.ndarray): Normalized adjacency matrix of the network.

    Returns:
    numpy.ndarray: A vector of modal controllability values for each node.
    """
    A = Normalization(A)
    T, U = schur(A, output='real')  # Schur decomposition for stability
    eigVals = np.diag(T)
    N = A.shape[0]
    phi = np.zeros(N)
    for i in range(N):
        phi[i] = np.sum(U[i, :]**2 * (1 - eigVals**2))
    return phi

def PyC_control_metrics(df, col_name):
    """
    Calculate control metrics for a DataFrame containing adjacency matrices.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data.
    col_name (str): Column name in df containing adjacency matrices.

    Returns:
    pandas.DataFrame: DataFrame with additional columns for control metrics.
    """
    # Normalization
    df['A_Norm'] = df[col_name].apply(Normalization)

    # Calculating control metrics
    df['Average'] = df[col_name].apply(PyC_AverageControl)
    df['Modal'] = df[col_name].apply(PyC_ModalControl)
    df['TimeConstant'] = df['A_Norm'].apply(lambda x: np.diag(x))

    return df

# Example usage:
# df = pd.read_csv('your_data.csv') # Load your data
# df = PyC_control_metrics(df_AB, 'A_matrice')
# print(df.head())

def PyC_melted(df, col_name, index, group_column = 'Group', entity_ids = 'user_id' ):
    """
    Transforms a DataFrame by melting it based on a specified column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be transformed.
    col_name (str): The name of the column to be melted.
    index (list): The new column names for the melted DataFrame.
    group_column (str, optional): The name of the column representing groups. Defaults to 'Group'.
    entity_ids (str, optional): The name of the column representing entity IDs. Defaults to 'user_id'.

    Returns:
    pandas.DataFrame: A melted DataFrame.
    """
    # Create a new DataFrame based on the specified column
    df_control = pd.DataFrame(df[col_name].tolist(), columns=index)
    # Concatenate with the 'group_column' and 'entity_ids' columns from the original DataFrame
    df_control = pd.concat([df[[ group_column, entity_ids]], df_control], axis=1)
    # Melt the DataFrame
    df_melted = pd.melt(df_control, id_vars=[group_column], value_vars=index, var_name='Index')
    return df_melted


def PyC_stat_tests(df_melted, value='value', group_column='Group'):
    """
    Perform statistical tests on the provided DataFrame.

    Parameters:
    df_melted (pandas.DataFrame): DataFrame containing the data for testing.
    value (str, optional): Column name of the values to be tested. Defaults to 'value'.
    group_column (str, optional): Column name containing group names. Defaults to 'Group'.

    Returns:
    pandas.DataFrame: DataFrame containing the results of the statistical tests.
    """
    index = df_melted['Index'].unique()
    groups = df_melted[group_column].unique()
    
    if len(groups) != 2:
        raise ValueError("The DataFrame must contain exactly two unique groups for comparison.")

    group1, group2 = groups
    normality_results = pd.DataFrame(index=index, columns=['g1_Normal', 'g2_Normal'])
    test_results = pd.DataFrame(index=index, columns=['Test_Stat', 'P_Value', 'Test_Type'])

    for idx in index:
        group_data = df_melted[df_melted['Index'] == idx]
        group_1 = group_data[group_data[group_column] == group1][value]
        group_2 = group_data[group_data[group_column] == group2][value]

        # Shapiro-Wilk test for normality
        normality_results.loc[idx, 'g1_Normal'] = stats.shapiro(group_1).pvalue > 0.05
        normality_results.loc[idx, 'g2_Normal'] = stats.shapiro(group_2).pvalue > 0.05

        # Choose statistical test based on normality results
        if normality_results.loc[idx, 'g1_Normal'] and normality_results.loc[idx, 'g2_Normal']:
            # T-test for normally distributed groups
            stat, p = stats.ttest_ind(group_1, group_2, nan_policy='omit')
            test_results.loc[idx] = [stat, p, 't-test']
        else:
            # Mann-Whitney U test for non-normally distributed groups
            stat, p = stats.mannwhitneyu(group_1, group_2, alternative='two-sided')
            test_results.loc[idx] = [stat, p, 'Mann-Whitney']

    # Bonferroni correction for multiple testing
    test_results['Adjusted_P_Value'] = sm.stats.multipletests(test_results['P_Value'], method='bonferroni')[1]
    test_results['Is_Significant'] = test_results['Adjusted_P_Value'] < 0.05

    # Merge normality and test results
    stat_results = normality_results.merge(test_results, left_index=True, right_index=True)

    return stat_results

def PyC_comparison_plot(df, group_column, result_stat_test, y_label='Y Axis', name_fig='file.jpg'):
    """
    Creates a group comparison plot based on statistical test results.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data for plotting.
    group_column (str): Column name in df that contains group names.
    result_stat_test (pandas.DataFrame): DataFrame containing the results of statistical tests.
    y_label (str, optional): Label for the Y-axis. Defaults to 'Y Axis'.
    name_fig (str, optional): Filename for saving the figure. Defaults to 'file.jpg'.

    Returns:
    None: The function generates and saves a plot.
    """
    # Determine the unique groups
    unique_groups = df[group_column].unique()
    if len(unique_groups) != 2:
        raise ValueError("The DataFrame must contain exactly two unique groups for comparison.")

    # Calculate median values for plotting
    median_values_df = df.groupby([group_column, 'Index'])['value'].median().reset_index()
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(20, 8), dpi=100)
    sns.barplot(x='Index', y='value', hue=group_column, data=median_values_df,
                hue_order=unique_groups, palette={unique_groups[0]: 'gray', unique_groups[1]: 'blue'}, ax=ax)

    # Annotate significant results
    for i, index in enumerate(df['Index'].unique()):
        if result_stat_test.loc[index, 'Is_Significant']:
            ax.get_xticklabels()[i].set_weight('bold')

    # Annotation configuration
    pairs = [((index, unique_groups[0]), (index, unique_groups[1])) for index in df['Index'].unique()]
    annotator = Annotator(ax, pairs, data=df, x='Index', y='value', hue=group_column)
    num_comparisons = len(df['Index'].unique())
    pvalue_thresholds = [(1.0e-03 / num_comparisons, '***'), (1.00e-02 / num_comparisons, '**'),
                        (5.00e-02 / num_comparisons, '*'), (1, 'ns')]
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', pvalue_thresholds=pvalue_thresholds)
    annotator.apply_test().annotate()

    # Customize plot appearance
    ax.set_xlabel('')
    ax.set_ylabel(y_label, fontweight='bold', fontsize=28, labelpad=30)
    leg = ax.legend(fontsize='22', loc='lower left', frameon=False)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=26, labelcolor='black', width=2.5)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    sns.despine(top=True, right=True)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color('black')

    plt.tight_layout()
    plt.show()
    fig.savefig(name_fig, bbox_inches='tight', dpi=300)

# Example usage:
# df_melted = PyC_melted(df, 'Average', 'Index')
# result_stat_test = PyC_stat_tests(df_melted)
# PyC_comparison_plot(df_melted, result_stat_test, 'Y Axis Label', 'output_figure.jpg')

