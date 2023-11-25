# PyC_ControlMetrics


## Overview
"PyC_ControlMetrics" is a Python repository focused on computing key control metrics in network theory, particularly for complex systems like neural and social networks. It includes functions for calculating Average Controllability, Modal Controllability, and Time Constants of nodes within a network.

## Background
Controllability in networks is crucial for understanding how individual nodes or elements influence the overall dynamics of a system. This repository provides tools to measure:
- **Average Controllability (AC):** Assessing how easily the state of a node can be steered using input controls.
- **Modal Controllability (MC):** Indicating a node's ability to move the system into various states.
- **Time Constants:** Evaluating the system's response speed.

## Installation
Clone this repository and install the required packages using:
```bash
git clone https://github.com/PsyControl/PyC_ControlMetrics.git
cd PyC_ControlMetrics
pip install numpy pandas scipy
```

## Usage
The repository includes functions for controllability analysis:

- **PyC_AverageControl:** Computes AC for each network node.
- **PyC_ModalControl:** Determines MC for each node.
- **Time Constant** calculation using normalized adjacency matrices.

```python

import pandas as pd
from PyC_ControlMetrics import PyC_AverageControl, PyC_ModalControl, Normalization

# Assuming your DataFrame 'df' contains an 'A_matrice' column with adjacency matrices
df['A_Norm'] = df['A_matrice'].apply(Normalization)
df['Average'] = df['A_matrice'].apply(PyC_AverageControl)
df['Modal'] = df['A_matrice'].apply(PyC_ModalControl)
df['TimeConstant'] = df['A_Norm'].apply(np.diag)
```
