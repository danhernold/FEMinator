import numpy as np
import numba
from typing import Callable
import matplotlib.pyplot as plt
import sys


def example_problem():
    side_nodes = 4
    n_nodes = side_nodes * side_nodes
    x_nodes = np.linspace(0, 1, side_nodes)
    y_nodes = np.linspace(0, 1, side_nodes)

    nodes = np.concatenate(
        [_.reshape(-1, 1) for _ in np.meshgrid(y_nodes, x_nodes)], axis=-1)

    print(nodes.shape)

    domain_size = 300
    x_domain = np.linspace(0.0, 1.0, domain_size)
    y_domain = np.linspace(0.0, 1.0, domain_size)

    nodes_on_edge = np.array(
        [i for i, node in enumerate(nodes) if (0 in node or 1 in node)])
    nodes_on_domain = np.array(
        [i for i, node in enumerate(nodes) if not (0 in node or 1 in node)])

    domain = np.concatenate(
        [_.reshape(-1, 1) for _ in np.meshgrid(x_domain, y_domain)], axis=1)

    edge = np.concatenate([
        np.concatenate([
            np.linspace(0., 1, domain_size).reshape(-1, 1),
            np.zeros(domain_size).reshape(-1, 1)
        ],
                       axis=1),
        np.concatenate([
            np.linspace(0., 1, domain_size).reshape(-1, 1),
            np.ones(domain_size).reshape(-1, 1)
        ],
                       axis=1),
        np.concatenate([
            np.zeros(domain_size).reshape(-1, 1),
            np.linspace(0., 1, domain_size).reshape(-1, 1)
        ],
                       axis=1),
        np.concatenate([
            np.ones(domain_size).reshape(-1, 1),
            np.linspace(0., 1, domain_size).reshape(-1, 1)
        ],
                       axis=1)
    ])

    B = np.zeros(nodes.shape[0])
    B[np.all(nodes == np.array([0.0, 0.0]), axis=1)] = 1.0
    B[np.all(nodes == np.array([1.0, 0.0]), axis=1)] = 1.0
    B[np.all(nodes == np.array([0.0, 1.0]), axis=1)] = 1.0
    B[np.all(nodes == np.array([1.0, 1.0]), axis=1)] = 1.0

    width = 0.25

    return nodes_on_edge, nodes_on_domain, nodes, domain, edge, B, width


@numba.jit(nopython=True, fastmath=True)
def gaussian_basis(x: np.ndarray, basis: np.ndarray, width: float):
    return np.exp(-np.sqrt(np.square(x - basis).sum()) / width)


@numba.jit(nopython=True, fastmath=True)
def linear_basis(x: np.ndarray, basis: np.ndarray, width: float):
    return np.maximum(1 - np.abs(x - basis).sum() / width, 0)


basis = linear_basis


@numba.jit(nopython=True, fastmath=True)
def integrate(nodes_on_edge: np.ndarray, nodes_on_domain: np.ndarray,
              nodes: np.ndarray, edge: np.ndarray, domain: np.ndarray,
              B: np.ndarray, width: float):
    """Integrate over domain to attain K and F.
    
    K_ji = integral (\nabla \dot basis_i) * basis_j
    F_j = -integral basis_j * B
    """
    num_nodes, dim = nodes.shape
    num_domain, _ = domain.shape
    num_edge, _ = edge.shape

    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros(num_nodes)

    epsilon = 1e-6
    dx = np.array([epsilon, 0])
    dy = np.array([0, epsilon])

    integral_discretization_domain = num_domain 
    integral_discretization_edge = num_edge 

    # Integrate over domain
    for j in nodes_on_domain:
        for i in nodes_on_domain:

            # Integrate over domain
            for p in range(num_domain):
                basis_j = basis(domain[p], nodes[j], width)
                basis_i = basis(domain[p], nodes[i], width)

                dx_basis_j = (basis(domain[p] + dx, nodes[j], width) -
                              basis_j) / epsilon
                dy_basis_j = (basis(domain[p] + dy, nodes[j], width) -
                              basis_j) / epsilon

                K[j, i] += (basis_i * (dx_basis_j + dy_basis_j) \
                    / integral_discretization_domain)

    # Integrate over edge
    for j in range(num_nodes):
        for i in range(num_nodes):
            for p in range(num_edge):
                basis_j = basis(edge[p], nodes[j], width)
                basis_i = basis(edge[p], nodes[i], width)
                K[j, i] += -2*(basis_i * basis_j / integral_discretization_edge)

    # Integrate over domain
    for j in range(num_nodes):
        for p in range(num_domain):
            basis_j = basis(domain[p], nodes[j], width)
            F[j] += -(basis_j * B[j] / integral_discretization_domain)

    return K, F


def calculate_stress(x: np.ndarray, nodes: np.ndarray, u: np.ndarray,
                     width: float):
    """Calculate stress in a point given the solution u.

    stress(x) = sum_N u_i * basis_i(x)
    """

    num_samples, _ = x.shape
    num_nodes, _ = nodes.shape

    samples_stress = []
    for i in range(num_samples):
        sample_stress = 0.0
        for j in range(num_nodes):
            sample_stress += basis(x[i], nodes[j], width) * u[j]

        samples_stress.append(sample_stress)

    return np.array(samples_stress)


def verify_that_diffeq_holds(x, nodes, u, b, width):
    num_samples, _ = x.shape
    num_nodes, _ = nodes.shape

    epsilon = 1e-5
    dx = np.array([epsilon, 0])
    dy = np.array([0, epsilon])

    for i in range(num_samples):

        stress = 0.0
        stress_dx = 0.0
        stress_dy = 0.0

        for j in range(num_nodes):
            stress += basis(x[i], nodes[j], width) * u[j]
            stress_dx += basis(x[i] + dx, nodes[j], width) * u[j]
            stress_dy += basis(x[i] + dy, nodes[j], width) * u[j]

        divergence = ((stress_dx - stress) + (stress_dy - stress)) / epsilon

        print("Divergence", divergence, "B", b[i], "Divergence + B",
              divergence + b[i])


nodes_on_edge, nodes_on_domain, nodes, domain, edge, B, width = example_problem(
)
K, F = integrate(nodes_on_edge, nodes_on_domain, nodes, edge, domain, B, width)

u = np.linalg.solve(K, F)

print()
print("Proof that it works -- if divergence + B == 0 then the equation holds")
verify_that_diffeq_holds(nodes, nodes, u, B, width)
print(" ------------------ ")
print()
print()

s = calculate_stress(domain, nodes, u, width)

domain_size, _ = domain.shape
domain_size = int(np.sqrt(domain_size))

plt.figure(figsize=(20, 10))
plt.imshow(s.reshape(domain_size, domain_size), origin="lower")
plt.colorbar()
plt.yticks([])
plt.xticks([])
plt.show()
