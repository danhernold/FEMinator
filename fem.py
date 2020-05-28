import numpy as np
from typing import Callable

nodes = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
element = np.array([[0, 1, 2, 3]])

basis_functions = [lambda x, y: (1 - x) * (1 - y) / 4, lambda x, y: (1 - x) * (1 + y) / 4,
                   lambda x, y: (1 + x) * (1 + y) / 4, lambda x, y: (1 + x) * (1 - y) / 4]

strain_interpolation_matrix = np.array([
    [[lambda y: (1 + y) / 4, 0, lambda y: 0, 0, lambda y: 0], [0, lambda x: (1 + x) / 40]],
])

# integral basis_i * basis_j

def jacobian(nodes: np.ndarray, element: np.ndarray):
    num_nodes, dim_nodes = nodes.shape()
    return dim_nodes


def integrate(basis_functions: Callable[[np.ndarray], float]):
    # first_element = elements[0]

    # first_element_nodes = []
    # for node in first_element:
    #    first_element_nodes.append(nodes[node])
    pass
