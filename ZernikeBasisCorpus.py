import numpy as np
from scipy.special import eval_jacobi
from scipy.integrate import trapezoid

class ZernikeBasisCorpus:
    def __init__(self, n_max=20, res=300, lam=0.6):
        self.n_max = n_max
        self.res = res
        self.lam = lam
        self._prepare_polar_grid()
        self._build_basis()

    def _prepare_polar_grid(self):
        self.rho   = np.linspace(0.0, 1.0, self.res).astype(np.float64)
        self.theta = np.linspace(0.0, 2*np.pi, self.res).astype(np.float64)
        self.r, self.t = np.meshgrid(self.rho, self.theta, indexing="xy")
        self.area_weights = self.r

    def _zernike_radial(self, n, m, r):
        m = abs(m)
        k = (n - m) // 2
        
        return (r**m) * eval_jacobi(k, 0, m, 2*(r**2) - 1)

    def _one_zernike(self, n, m):
        R = self._zernike_radial(n, m, self.r)
        Z = R * np.exp(1j * m * self.t)
        integrand = (np.abs(Z) ** 2) * self.area_weights
        norm = trapezoid(trapezoid(integrand, self.rho, axis=1), self.theta, axis=0)
        Z /= np.sqrt(norm)

        return Z.astype(np.complex128)

    def _build_basis(self):
        self.basis = []
        self.indices = []
        for n in range(self.n_max + 1):
            for m in range(-n, n + 1, 2):
                if (n - abs(m)) % 2 == 0:
                    self.basis.append(self._one_zernike(n, m))
                    self.indices.append((n, m))
        self.idx_map = {idx: i for i, idx in enumerate(self.indices)}
        self.radial_order = np.argsort([n for (n, m) in self.indices])

        # self.check_orthogonality()

    def get_indices(self):
        return self.indices
    
    def get_basis(self):
        return self.basis

    def check_orthogonality(self):
        all_passed = True
        for i, B1 in enumerate(self.basis):
            for j, B2 in enumerate(self.basis):
                if j < i: continue
                inner = B1.conj() * B2 * self.area_weights
                val = trapezoid(trapezoid(inner.real, self.rho, axis=1), self.theta, axis=0)
                expected = 1.0 if i == j else 0.0
                if not np.isclose(val, expected, atol=1e-1):
                    print(f"Not orthogonal: ({self.indices[i]}) · ({self.indices[j]}) = {val}")
                    all_passed = False
        if all_passed:
            print("All Zernike basis functions passed the orthogonality check!")

    def project(self, mask):
        coeffs = np.zeros(len(self.basis), dtype=np.complex128)
        for i, B in enumerate(self.basis):
            integrand = mask * np.conj(B) * self.area_weights
            coeffs[i] = trapezoid(trapezoid(integrand, self.rho, axis=1), self.theta, axis=0)

        return coeffs

    def synthesize(self, coeffs):
        # inverse of project function
        out = np.zeros((self.res, self.res), dtype=np.complex128)
        for c, B in zip(coeffs, self.basis):
            out += c * B

        return np.real(out)

    def radial_freqprop(self, coeffs, lam=None):
        """radial frequency propagation:"""
        if lam is None: lam = self.lam
        z_raw = np.asarray(coeffs, dtype=np.complex128)
        z_mod = z_raw.copy()
        for i in self.radial_order:
            n, m = self.indices[i]
            src_idx = self.idx_map.get((n - 2, m), None)
            if src_idx is None:
                continue
            src = z_mod[src_idx]  # cascaded-source
            if np.abs(src) > 0.0:
                z_mod[i] += lam * src

        return z_mod

    def invert_radialfreqprop(self, coeffs_mod, lam=None):
        if lam is None: lam = self.lam
        z_mod = np.asarray(coeffs_mod, dtype=np.complex128)
        z_rec = z_mod.copy()
        for i in self.radial_order:
            n, m = self.indices[i]
            src_idx = self.idx_map.get((n - 2, m), None)
            if src_idx is None:
                continue
            src = z_mod[src_idx]  # use MOD source (cascaded policy)
            if np.abs(src) > 0.0:
                z_rec[i] -= lam * src
                
        return z_rec
    
    def invert_angularfreqprop(self, coeffs_mod, lam=None):
        if lam is None: lam = self.lam
        z_mod = np.asarray(coeffs_mod, dtype=np.complex128)
        z_rec = z_mod.copy()
        for i in self.radial_order:
            n, m = self.indices[i]
            if m >= 2:
                src_idx = self.idx_map.get((n, m - 2), None)
            else:
                continue
            if src_idx is None:
                continue
            src = z_mod[src_idx]  # use MOD source (cascaded policy)
            if np.abs(src) > 0.0:
                z_rec[i] -= lam * src

        #run conjugate symmetry for m < 0
        for i in range(len(self.basis)):
            n, m = self.indices[i]
            if m < 0:
                pos_idx = self.idx_map.get((n, -m), None)
                if pos_idx is not None:
                    z_rec[i] = np.conj(z_rec[pos_idx])

        return z_rec
    
    def angular_freqprop(self, coeffs, lam=None):
        """angular frequency propagation, we just propagate on m >=0 indices,
        and use conjugate symmetry for m < 0 indices."""
        if lam is None: lam = self.lam
        z_raw = np.asarray(coeffs, dtype=np.complex128)
        z_mod = z_raw.copy()
        for i in range(len(self.basis)):
            n, m = self.indices[i]
            if m >= 2:
                src_idx = self.idx_map.get((n, m - 2), None)
            else:
                continue
            src_idx = self.idx_map.get((n, m - 2), None)
            if src_idx is None:
                continue
            src = z_mod[src_idx]  # cascaded-source
            if np.abs(src) > 0.0:
                z_mod[i] += lam * src

        #run conjugate symmetry for m < 0
        for i in range(len(self.basis)):
            n, m = self.indices[i]
            if m < 0:
                pos_idx = self.idx_map.get((n, -m), None)
                if pos_idx is not None:
                    z_mod[i] = np.conj(z_mod[pos_idx])

        return z_mod

    def encode(self, mask, lam=None, return_raw=False):
        raw = self.project(mask)
        mod_rfreqprop = self.radial_freqprop(raw, lam=lam)
        mod_arfreqprop = self.angular_freqprop(mod_rfreqprop, lam=lam)

        if return_raw:
            return raw, mod_arfreqprop
        else:
            return mod_arfreqprop
    
    def decode(self, coeffs_mod, lam=None, decode_freqprop=True):
        if decode_freqprop:
            raw_rec = self.invert_angularfreqprop(coeffs_mod, lam=lam)
            raw_rec = self.invert_radialfreqprop(raw_rec, lam=lam)
        else:
            raw_rec = coeffs_mod
        
        recon = self.synthesize(raw_rec)

        return raw_rec, recon