# Network Control Metrics

This repository contains Python code for calculating control metrics in complex networks, specifically focusing on neural networks. The code computes Average and Modal Controllability metrics based on the adjacency matrices of these networks.

## Features
- Normalization of adjacency matrices.
- Calculation of Average Controllability.
- Calculation of Modal Controllability.
- Extraction of time constants from normalized adjacency matrices.

## Usage
The main functions are `PyC_AverageControl` and `PyC_ModalControl`, which take a normalized adjacency matrix as input and return control metrics for each node in the network.

## Dependencies
- numpy
- scipy

## References
The methodologies implemented here are based on the work done by the Bassett Lab, University of Pennsylvania, and can be found in the following paper: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia, Vettel, Miller, Grafton & Bassett, Nature Communications 6:8414, 2015.
