import numpy as np
import numba
from typing import Callable
import matplotlib.pyplot as plt
import sys

nodes = np.array([
    [0.0, 0.0],
    [0.0, 0.25],
    [0., 0.5],
    [0.0, 0.75],
    [0.0, 1.0],
    [0.25, 0.0],
    [0.25, 0.25],
    [0.25, 0.5],
    [0.25, 0.75],
    [0.25, 1.0],
    [0.5, 0.0],
    [0.5, 0.25],
    [0.5, 0.5],
    [0.5, 0.75],
    [0.5, 1.0],
    [0.75, 0.0],
    [0.75, 0.25],
    [0.75, 0.5],
    [0.75, 0.75],
    [0.75, 1.0],
    [1.0, 0.0],
    [1.0, 0.25],
    [1.0, 0.5],
    [1.0, 0.75],
    [1.0, 1.0],
])

domain_size = 200
x_domain = np.linspace(0., 1., domain_size)
y_domain = np.linspace(0., 1., domain_size)

domain = np.concatenate(
    [_.reshape(-1, 1) for _ in np.meshgrid(x_domain, y_domain)], axis=1)

B = 1.0
width = 4.0

print("domain shape", domain.shape)


@numba.jit(nopython=True, fastmath=True)
def gaussian_basis(x: np.ndarray, basis: np.ndarray, width: float):
    return np.exp(-np.sqrt(np.square(x - basis).sum()) / width)


@numba.jit(nopython=True, fastmath=True)
def linear_basis(x: np.ndarray, basis: np.ndarray, width: float):
    return np.maximum(1 - np.abs(x - basis).sum() / width, 0)


@numba.jit(nopython=True, fastmath=True)
def integrate(nodes: np.ndarray, domain: np.ndarray, B: float, width: float):
    """Integrate over domain to attain K and F.
    
    K_ji = integral (\nabla \dot basis_i) * basis_j
    F_j = -integral basis_j * B
    """
    num_nodes, dim = nodes.shape
    num_domain, _ = domain.shape

    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros(num_nodes)

    epsilon = 1e-2
    dx = np.array([epsilon, 0])
    dy = np.array([0, epsilon])

    C = 0

    integral_discretization_sz = domain_size * domain_size
    for j in range(num_nodes):
        for i in range(num_nodes):
            for p in range(num_domain):
                basis_j = linear_basis(domain[p], nodes[j], width)
                basis_i = linear_basis(domain[p], nodes[i], width)

                dx_basis_i = (linear_basis(domain[p] + dx, nodes[i], width) -
                              basis_i) / epsilon
                dy_basis_i = (linear_basis(domain[p] + dy, nodes[i], width) -
                              basis_i) / epsilon

                K[j, i] += (dx_basis_i + dy_basis_i) * basis_j\
                    / integral_discretization_sz

        for p in range(num_domain):
            basis_j = linear_basis(domain[p], nodes[j], width)
            F[j] += -(basis_j * B / integral_discretization_sz)

    return K, F


def strain(x: np.ndarray, nodes: np.ndarray, u: np.ndarray, width: float):
    """Calculate stress in a point given the solution u.

    stress(x) = sum_N u_i * basis_i(x)
    """

    num_samples, _ = x.shape
    num_nodes, _ = nodes.shape

    samples_stress = []
    for i in range(num_samples):
        sample_stress = 0.0
        for j in range(num_nodes):
            sample_stress += linear_basis(x[i], nodes[j], width) * u[j]

        samples_stress.append(sample_stress)

    return np.array(samples_stress)


K, F = integrate(nodes, domain, B, width)
print("K", K)
print("F", F)

# Force bottom left to be 5.0
F[0] = 5.0

# Diagonalize K at 2
K[0, :] = 0.0
K[0, 0] = 1.0

u = np.linalg.solve(K, F)


def diff_hold(x, nodes, u, b, width):
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
            stress += linear_basis(x[i], nodes[j], width) * u[j]
            stress_dx += linear_basis(x[i] + dx, nodes[j], width) * u[j]
            stress_dy += linear_basis(x[i] + dy, nodes[j], width) * u[j]

        divergence = ((stress_dx - stress) + (stress_dy - stress)) / epsilon

        print("Divergence", divergence, "B", b, "Divergence + B",
              divergence + b)


print()
print("Proof that it works -- if divergence + B == 0 then the equation holds")
diff_hold(nodes, nodes, u, B, width)
print(" ------------------ ")
print()
print()

print("u", u)
print("stress", strain(nodes, nodes, u, width))
print("k @ u", K @ u)

s = strain(domain, nodes, u, width)

plt.figure(figsize=(20, 10))
plt.imshow(s.reshape(domain_size, domain_size), origin="lower")
plt.colorbar()
plt.yticks([])
plt.xticks([])
plt.show()
