import numpy as np
import sys
sys.path.append("utils/")
import ZernikeBasisCorpus
import MaskConvert

class ShapeGeometryEncoder:
    def __init__(self, n_max=20, res=256, lam=0.6, encode_len=128):
        assert n_max >= 0, "n_max must be non-negative"
        self.encode_len = encode_len
        if encode_len is not None:
            encode_len_tmp = 0
            for n_tmp in range(encode_len):
                encode_len_tmp += int(np.ceil(float(n_tmp)/2. + 0.5))*2
                if encode_len_tmp >= encode_len:
                    self.n_max = n_tmp
                    break
        else:
            self.n_max = n_max

        self.res = res
        self.lam = lam

        self.zernike_corpus = ZernikeBasisCorpus.ZernikeBasisCorpus(n_max=self.n_max, res=self.res, lam=self.lam)

    def compute_zernike_coeffs(self, shape_mask):
        coeffs = self.zernike_corpus.project(shape_mask)

        return coeffs

    def get_real_compact_coeffs(self, coeffs):
        m_n_indices = self.zernike_corpus.get_indices()
        real_coeffs = []
        for indice_tmp, coeff_complex in zip(m_n_indices, coeffs):
            n, m = indice_tmp
            if m >= 0:
                real_coeffs.append(float(np.real(coeff_complex)))
                real_coeffs.append(float(np.imag(coeff_complex)))

        return real_coeffs
    
    def index_coeffs_from_realcompact_coeffs(self, m, n, real_coeffs):
        m_n_indices = self.zernike_corpus.get_indices()
        if m < 0:
            m = -1 * m

        idx = 0
        for indice_tmp in m_n_indices:
            n_tmp, m_tmp = indice_tmp
            if m_tmp == m and n_tmp == n:
                real_part = real_coeffs[idx]
                imag_part = real_coeffs[idx + 1]

                return real_part, imag_part
            else:
                if m_tmp >= 0:
                    idx += 2
        
        raise ValueError(f"Index (m={m}, n={n}) not found in Zernike basis indices.")

    def inverse_real_compact_coeffs(self, real_coeffs):
        m_n_indices = self.zernike_corpus.get_indices()
        coeffs = []
        for indice_tmp in m_n_indices:
            n, m = indice_tmp
            real_part, imag_part = self.index_coeffs_from_realcompact_coeffs(m, n, real_coeffs)
            if m >= 0:
                coeff_complex = np.complex128(complex(real_part, imag_part))
            else:
                coeff_complex = np.conj(np.complex128(complex(real_part, imag_part)))

            coeffs.append(coeff_complex)

        return coeffs

    def encode(self, shape_mask, mask_in_euclidean=False, return_raw_coeffs=False):
        # the shape mask has to be in polar coordinates
        if mask_in_euclidean:
            shape_mask = MaskConvert.euclidean_mask_to_polar(shape_mask, 
                                                             theta_resolution=self.res, 
                                                             rho_resolution=self.res)
        raw_complex, freqprop_coeff_complex = self.zernike_corpus.encode(shape_mask, lam=self.lam, return_raw=True)
        real_coeffs = self.get_real_compact_coeffs(freqprop_coeff_complex)

        if return_raw_coeffs:
            return real_coeffs, freqprop_coeff_complex, raw_complex

        return real_coeffs

    def decode(self, coeffs):
        coeffs_complex = self.inverse_real_compact_coeffs(coeffs)
        shape_rec = self.zernike_corpus.decode(coeffs_mod=coeffs_complex, lam=self.lam, decode_freqprop=True)[1]

        return shape_rec