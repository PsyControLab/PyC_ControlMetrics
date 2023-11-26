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


## Installation
Clone this repository and install the required packages using:
```bash
git clone https://github.com/PsyControl/PyC_ControlMetrics.git
cd PyC_ControlMetrics
pip install numpy scipy pandas matplotlib seaborn
```

## Usage
The repository includes functions for controllability analysis:

- **PyC_AverageControl:** Computes AC for each network node.
- **PyC_ModalControl:** Determines MC for each node.
- **Time Constant** calculation using normalized adjacency matrices.

```python

import pandas as pd
from PyC_ControlMetrics import *

# Assuming your DataFrame 'df' contains an 'A_matrice' column with adjacency matrices
df['A_Norm'] = df['A_matrice'].apply(Normalization)
df['Average'] = df['A_matrice'].apply(PyC_AverageControl)
df['Modal'] = df['A_matrice'].apply(PyC_ModalControl)
df['TimeConstant'] = df['A_Norm'].apply(np.diag)
```

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
