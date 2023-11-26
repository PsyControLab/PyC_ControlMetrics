# Network Control Metrics: Average and Modal Controllability

This repository contains Python functions for calculating Average Controllability (AC) and Modal Controllability (MC) in networks. These metrics are crucial in network theory, particularly for understanding the dynamics and control mechanisms in complex systems like neural or social networks.

## Overview

### Average Controllability (AC)

- **Definition**: Measures the ease with which the state of a node can be altered using input controls, reflecting the influence of individual nodes within a network.
- **Importance**: Crucial for understanding how interventions can change the state of a node and the overall network.
- **Mathematical Formulation**: AC is defined as `AC_j = trace(∑_i=0^∞ A^i B_j B_j^T (A^T)^i)`, where `A` is the network's adjacency matrix and `B_j` is the jth canonical vector.
- **Application**: In emotional dynamics studies, AC can indicate how easily an emotion (node) can be influenced or controlled.

### Modal Controllability (MC)

- **Definition**: Quantifies a node's ability to drive the system into different states, focusing on controlling individual modes of the system.
- **Importance**: Essential for understanding the network's potential variability and capacity to reach various states or modes.
- **Mathematical Formulation**: MC is calculated as `MC_j = ∑_i=1^n [1 - ξ_i^2(A)] v_ji^2`, where `ξ_i(A)` and `v_ji` are the eigenvalues and eigenvectors of `A`, respectively.
- **Application**: Represents the ability to drive an individual into specific emotional states.

### Conceptual Differences

- AC is associated with averaged interconnections between nodes, while MC relates to steering the system into its various possible modes.

### Time Constant (τ)

- **Definition**: Measures the system's speed of response, defined as the inverse of the system's eigenvalues.


## Installation
To use these functions, you need to have Python installed on your system along with the following libraries:
Clone this repository and install the required packages using:
```bash
git clone https://github.com/PsyControl/PyC_ControlMetrics.git
cd PyC_ControlMetrics
pip install numpy scipy pandas matplotlib seaborn
```
## Usage

This section demonstrates how to use the provided Python functions to calculate and analyze network control metrics. Ensure you have the necessary libraries installed and the `PyC_ControlMetrics` module imported.

- **PyC_AverageControl:** Computes AC for each network node.
- **PyC_ModalControl:** Determines MC for each node.
- **Time Constant** calculation using normalized adjacency matrices.

To utilize the network control metrics functions in your projects, follow these steps:

1. **Import Required Libraries**: Import the necessary Python libraries for data manipulation, statistical analysis, and visualization.

    ```python
    import pandas as pd
    import numpy as np
    from scipy.linalg import svdvals, schur
    from scipy import stats
    import statsmodels.api as sm
    import seaborn as sns
    import matplotlib.pyplot as plt
    from statannotations.Annotator import Annotator
    from PyC_ControlMetrics import *  # Import custom functions for control metrics
    ```

2. **Load Your Network's Adjacency Matrix**: This matrix represents the network you are analyzing.

    ```python
    # Example: Loading an adjacency matrix from a CSV file
    A = pd.read_csv('path_to_your_adjacency_matrix.csv').values
    ```

3. **Prepare Your Data**: Assuming you have a DataFrame `df` that contains an 'A_matrice' column with adjacency matrices.

    ```python
    # Normalizing the adjacency matrices and calculating control metrics
    df['A_Norm'] = df['A_matrice'].apply(Normalization)
    df['Average'] = df['A_matrice'].apply(PyC_AverageControl)
    df['Modal'] = df['A_matrice'].apply(PyC_ModalControl)
    df['TimeConstant'] = df['A_Norm'].apply(np.diag)
    ```

4. **Calculate Control Metrics**: Utilize the `PyC_control_metrics` function to compute control metrics for your data.

    ```python
    # Calculating control metrics for the DataFrame
    df_control = PyC_control_metrics(df, 'A_matrice')
    print(df_control.shape)
    df_control.head()
    ```

5. **Transform Data for Statistical Analysis**: Melt the DataFrame to prepare it for statistical tests.

    ```python
    Index = []
    for i in range(len(df_control['A_matrice'][0])):
        Index.append(i+1)
    
    # Melting the DataFrame for statistical analysis
    df_melted = PyC_melted(df_control, 'TimeConstant', index='Index', group_column='Group', entity_ids='user_id')
    ```

6. **Perform Statistical Tests**: Conduct statistical tests on the melted DataFrame.

    ```python
    # Performing statistical tests on the melted DataFrame
    result_stat_test = PyC_stat_tests(df_melted, value='value', group_column='Group')
    ```

7. **Generate Comparison Plots**: Create plots to visually compare groups based on the statistical test results.

    ```python
    # Creating a comparison plot based on the statistical test results
    PyC_comparison_plot(df_melted, result_stat_test, 'Y Axis Label', 'output_figure.jpg')
    ```

8. **Analyze the Results**: The output from the above functions are vectors representing the controllability metrics for each node in your network. You can further analyze these results to gain insights into the network's dynamics.

    ```python
    # Example: Displaying the first five nodes' controllability metrics
    print("Average Controllability of first five nodes:", df_control['Average'].head())
    print("Modal Controllability of first five nodes:", df_control['Modal'].head())
    ```

By following these steps, you can effectively apply the network control metrics to your network data and gain valuable insights into the dynamics and control mechanisms of the network.


### Dependencies
- **Python**
- **NumPy**
- **SciPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
  
### Acknowledgments
Developed at Bassett Lab, University of Pennsylvania, 2016.

### References
Gu et al., Nature Communications, 6:8414, 2015.

```css
This R Markdown document is structured to provide a comprehensive overview of your project, including theoretical background, installation instructions, usage examples, and acknowledgments. You can adjust the content as needed to fit the specifics of your project.
```














