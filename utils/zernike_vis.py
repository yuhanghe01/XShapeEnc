import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def zernike_radial(n, m, r):
    R = np.zeros_like(r)
    for k in range((n - abs(m)) // 2 + 1):
        c = (-1)**k * factorial(n - k) / (
            factorial(k) *
            factorial((n + abs(m)) // 2 - k) *
            factorial((n - abs(m)) // 2 - k)
        )
        R += c * r**(n - 2 * k)
    return R

def zernike(n, m, r, theta):
    R = zernike_radial(n, m, r)
    return R * np.exp(1j * m * theta)

# Create polar grid
res = 300
rho = np.linspace(0, 1, res)
theta = np.linspace(0, 2*np.pi, res)
r, t = np.meshgrid(rho, theta)

# Convert polar to cartesian for plotting
x = r * np.cos(t)
y = r * np.sin(t)

import ZernikeBasisCorpus

zernike_basis_corpus = ZernikeBasisCorpus.ZernikeBasisCorpus(n_max=10, res=res)

indices = zernike_basis_corpus.get_indices()
basis = zernike_basis_corpus.get_basis()

def get_one_zernike_basis(n, m):
    idx = indices.index((n, m))
    return basis[idx]

# Plot several basis
fig, axes = plt.subplots(3, 4, figsize=(10, 10))
orders = [(0,0), (1,-1), (1,1), (2,-2), (2,0), (2,2), (3,-3), (3,-1), (3,1), (3,3)]
for ax, (n, m) in zip(axes.flatten(), orders):
    # Z = zernike(n, m, r, t)
    Z = get_one_zernike_basis(n, m)
    ax.pcolormesh(x, y, np.real(Z), cmap='RdBu', shading='auto')
    ax.set_title(f'Re[Z_{n}^{m}]')
    ax.set_aspect('equal')
    ax.axis('off')
plt.tight_layout()
plt.savefig('zernike_basis.png', dpi=300)
# plt.show()