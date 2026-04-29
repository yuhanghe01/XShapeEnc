import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.special import eval_jacobi

class ShapePoseEncoder:
    def __init__(self, res = 300, 
                 K = 6, 
                 L = 100, 
                 m_p = 10, 
                 seed_sigma = 0.1):
        """
        K: pose vector length
        L: encoding length
        m_p: angular frequency mode
        """
        self.K = K
        self.L = L
        self.m_p = m_p
        self.res = res
        self.seed_sigma = seed_sigma

        self.rho = np.linspace(0, 1, self.res)
        self.theta = np.linspace(0, 2 * np.pi, self.res)
        self.r, self.t = np.meshgrid(self.rho, self.theta)
        self._build_radial_window()
        self._build_zernike_radials()

    def _two_bump_gaussian(self, rho, c, sigma, offset, w1=0.5, w2=0.5):
        """
        Two-peak radial Gaussian mixture centered around c with sub-centers at c±offset.
        Guarantees two distinct maxima by enforcing offset >= sqrt(2)*sigma.
        Keeps both sub-centers inside [0,1] by shifting inward together if needed.
        """
        # Ensure true bimodality (two separate maxima for equal-variance Gaussians)
        min_offset = np.sqrt(2.0) * sigma
        if offset < min_offset:
            offset = min_offset

        c1, c2 = c - offset, c + offset

        # Shift both sub-centers inward together if either is out of [0,1]
        if c1 < 0.0:
            shift = -c1
            c1 += shift
            c2 += shift
        if c2 > 1.0:
            shift = c2 - 1.0
            c1 -= shift
            c2 -= shift

        # If still degenerate due to extreme c near edges, clamp and re-space minimally
        c1 = float(np.clip(c1, 0.0, 1.0))
        c2 = float(np.clip(c2, 0.0, 1.0))
        if abs(c2 - c1) < min_offset:
            mid = 0.5 * (c1 + c2)
            c1 = float(np.clip(mid - 0.5 * min_offset, 0.0, 1.0))
            c2 = float(np.clip(mid + 0.5 * min_offset, 0.0, 1.0))
            # If clamping reduced spacing, pull inward again
            if abs(c2 - c1) < min_offset:
                # Final fallback: place a two-peak pair around mid inside bounds
                span = min(min_offset, 0.5)  # keep safe
                c1 = float(np.clip(mid - 0.5 * span, 0.0, 1.0))
                c2 = float(np.clip(mid + 0.5 * span, 0.0, 1.0))

        g1 = np.exp(-((rho - c1) ** 2) / (2.0 * sigma ** 2))
        g2 = np.exp(-((rho - c2) ** 2) / (2.0 * sigma ** 2))

        return w1 * g1 + w2 * g2

    def check_orthonormal(self):
        K = self.K
        G = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                G[i, j] = trapezoid(self.windows[i] * self.windows[j] * self.rho, self.rho)
        print("Gram matrix (should be ~I):\n", G)
        print("max |off-diagonal|:", np.max(np.abs(G - np.eye(K))))
        print("min/max diag:", G.diagonal().min(), G.diagonal().max())

    def _build_radial_window(self, offset=None, weights=(0.5, 0.5), sigma=None):
        """
        Build two-bump Gaussian seed windows, then Gram–Schmidt under ⟨f,g⟩ = ∫ f g r dr.
        - offset: half-separation of the two bumps (default ~0.35*center spacing)
        - sigma: per-bump std (default self.seed_sigma)
        """
        centers = np.linspace(0.1, 0.9, self.K)
        if sigma is None:
            sigma = self.seed_sigma
        if offset is None:
            spacing = centers[1] - centers[0] if self.K > 1 else 0.2
            offset = 0.35 * spacing  # fairly separated by default

        w1, w2 = weights
        # --- build truly bimodal seeds ---
        self.seed_windows = [
            self._two_bump_gaussian(self.rho, c, sigma, offset, w1=w1, w2=w2)
            for c in centers
        ]

        self.windows = []

        for i in range(self.K):
            w = self.seed_windows[i].copy()
            for j in range(i):
                w -= self._radial_inner_product(w, self.windows[j]) * self.windows[j]
            for j in range(i):  # second pass
                w -= self._radial_inner_product(w, self.windows[j]) * self.windows[j]
            w /= np.sqrt(self._radial_inner_product(w, w)) + 1e-12
            self.windows.append(w)

    def _radial_inner_product(self, w1, w2):
        return trapezoid(w1 * w2 * self.rho, self.rho)

    def _build_zernike_radials(self):
        self.zernike_radials = list()
        self.valid_ns = list()

        n_min = max(0, self.m_p)
        for n in range(n_min, n_min + 2 * self.L + 100000):
            if (n - self.m_p) % 2 == 0 and n >= abs(self.m_p):
                self.valid_ns.append(n)
                self.zernike_radials.append(self._zernike_radial(n, self.m_p, self.rho))
            
            if len(self.zernike_radials) >= self.L:
                break

        assert len(self.zernike_radials) == self.L, f"Could not find enough Zernike radial functions for m_p={self.m_p}, L={self.L}."

        self.zernike_radials = np.array(self.zernike_radials)  # Shape: [L x res], L = K(K+1)/2
        self.radial_norms = np.array([trapezoid((R_n**2) * self.rho, self.rho) for R_n in self.zernike_radials])

    def _zernike_radial(self, n, m, r):
        m = abs(m)
        k = (n - m) // 2
     
        return (r**m) * eval_jacobi(k, 0, m, 2*(r**2) - 1)

    def inverse_pose_encoding(self, pos_coeffs):
        C = np.zeros((self.K, self.L), dtype=np.float64)
        for k in range(self.K):
            for n in range(self.L):
                integrand = self.windows[k] * self.zernike_radials[n] * self.rho
                C[k, n] = trapezoid(integrand, self.rho)

        pos_coeffs = np.asarray(pos_coeffs)
        scaled_coeffs = pos_coeffs * self.radial_norms

        p_recovered = scaled_coeffs @ np.linalg.pinv(C)

        return p_recovered
    
    def encode(self, p_vec, just_return_coeff=False):
        # pos_vec can be either a list or a numpy array, but it can just handle one pose vector at a time
        assert len(p_vec) == self.K, "Pose vector length must match number of radial windows K."
        radial_part = np.sum([p*w for p, w in zip(p_vec, self.windows)], axis=0)  # Shape: [res, res]
        f_pose = np.outer(radial_part, np.cos(self.m_p * self.theta)).T  # Shape: [res, res]
        coeffs = list()
        for R_n in self.zernike_radials:
            Znm = np.outer(R_n, np.cos(self.m_p * self.theta)).T  # Shape: [res, res]
            integrand = f_pose * Znm * self.r
            val = trapezoid(trapezoid(integrand, self.rho, axis=1), self.theta, axis=0)
            norm = trapezoid(trapezoid(Znm * Znm * self.r, self.rho, axis=1), self.theta, axis=0) + 1e-12
            coeffs.append(float(val / norm))

        if just_return_coeff:
            return coeffs
        else:
            p_recovered = self.inverse_pose_encoding(coeffs)
            return self.valid_ns, coeffs, f_pose, p_recovered

    def plot_pose_field(self, f_pose, filename='pose_field.pdf'):
        #convert to cartesian for visualization
        x = self.r * np.cos(self.t)
        y = self.r * np.sin(self.t)
        fig, ax = plt.subplots(figsize=(6,6))
        contour = ax.contourf(x, y, f_pose, levels=100, cmap ='RdBu_r')
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_ticks([-np.max(np.abs(f_pose)), 0, np.max(np.abs(f_pose))])
        cbar.ax.tick_params(labelsize=12)
        ax.set_aspect('equal')
        ax.set_xticks([])  
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()