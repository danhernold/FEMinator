import numpy as np
import numba
from typing import Callable
import matplotlib.pyplot as plt

nodes = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
elements = np.array([[0, 1, 2, 3]])

domain_size = 200
x_domain = np.linspace(0., 1., domain_size)
y_domain = np.linspace(0., 1., domain_size)

domain = np.concatenate(
    [_.reshape(-1, 1) for _ in np.meshgrid(x_domain, y_domain)], axis=1)

B = 1.0
width = 1.0


@numba.jit(nopython=True, fastmath=True)
def gaussian_basis(x: np.ndarray, basis: np.ndarray, width: float):
    return np.exp(-np.sqrt(np.square(x - basis).sum()) / width)

@numba.jit(nopython=True, fastmath=True)
def linear_basis(x: np.ndarray, basis: np.ndarray, width: float):
    return np.maximum(1 - np.sqrt(np.square(x - basis).sum()) / width, 0)


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

    epsilon = 1e-4
    dx = np.array([epsilon, 0])
    dy = np.array([0, epsilon])

    integral_discretization_sz = domain_size * domain_size
    for j in range(num_nodes):
        for i in range(num_nodes):
            for p in range(num_domain):
                basis_j = gaussian_basis(domain[p], nodes[j], width)
                basis_i = gaussian_basis(domain[p], nodes[i], width)

                dx_basis_i = (gaussian_basis(domain[p] + dx, nodes[i], width) -
                              basis_i) / epsilon
                dy_basis_i = (gaussian_basis(domain[p] + dy, nodes[i], width) -
                              basis_i) / epsilon

                K[j, i] += basis_i * basis_j\
                    / integral_discretization_sz

        for p in range(num_domain):
            basis_j = gaussian_basis(domain[p], nodes[j], width)
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
            sample_stress += gaussian_basis(x[i], nodes[j], width) * u[j]

        samples_stress.append(sample_stress)

    return np.array(samples_stress)


K, F = integrate(nodes, domain, B, width)
print("K", K)
print("F", F)

# Force topleft to be 1.0
F[2] = 1.0

# Diagonalize K at 2
K[2, :] = 0.0
K[2, 2] = 1.0

u = np.linalg.solve(K, F)


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
