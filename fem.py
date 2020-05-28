import numpy as np

nodes = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], [0.0, 1.0])
element = np.array([[0, 1, 2, 3]])


def jacobian(nodes: np.ndarray, element: np.ndarray):
    num_nodes, dim_nodes = nodes.shape()
    return dim_nodes

