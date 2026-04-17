import numpy as np
import sys
sys.path.append("utils/")
import ZernikeBasisCorpus
import MaskConvert
import plot_utils

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

def get_square_mask(res=300):
    # Resolution
    # Polar grid
    rho = np.linspace(0, 1, res)
    theta = np.linspace(0, 2 * np.pi, res)
    r, t = np.meshgrid(rho, theta)

    # Convert polar to cartesian
    x = r * np.cos(t)
    y = r * np.sin(t)
    # Define square within unit disk
    # Side length sqrt(2), i.e. half length = 1/√2 ~ 0.707 to fit in unit disk
    half_len = 1 / np.sqrt(2)
    square_mask = (np.abs(x) <= half_len) & (np.abs(y) <= half_len)

    show_mask_with_vector_edges(x, y, square_mask, "#0078D4")

    return square_mask


def show_mask_with_vector_edges(x, y, mask, color="#0078D4"):
    """
    x, y: same shape as mask, giving the Cartesian location of each polar cell
    mask: boolean or 0/1 array on the polar grid
    """
    z = mask.astype(float)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Fill the mask region smoothly on the (x, y) grid
    # Levels [0.5, 1.5] means "fill where z > 0.5"
    cf = ax.contourf(x, y, z, levels=[0.5, 1.5], colors=[color], antialiased=True)

    # Optional: draw the unit circle outline for context
    theta = np.linspace(0, 2*np.pi, 1024)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1.0, color="0.6", alpha=0.8, antialiased=True)

    # Optional: overlay the boundary as a crisp line in the same color
    ax.contour(x, y, z, levels=[0.5], colors=[color], linewidths=1.0, antialiased=True)

    ax.set_aspect('equal')
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("mask_with_vector_edges.png", bbox_inches="tight")
    plt.close()

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def plot_square_mask(color="#0078D4"):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Square that exactly fits in the unit disk: half side = 1/sqrt(2)
    half = 1 / np.sqrt(2)

    # Fill the square (vector graphics = crisp edges)
    ax.add_patch(Rectangle(
        (-half, -half), 2*half, 2*half,
        facecolor=color, edgecolor='none', antialiased=True
    ))

    # Optional: faint unit circle outline for context
    ax.add_patch(Circle((0, 0), radius=1, facecolor='none',
                        edgecolor='0.6', linewidth=1, antialiased=True))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Square Mask within Unit Disk", fontsize=14)
    plt.savefig("square_mask.png", bbox_inches='tight')
    plt.close(fig)

def main():
    res = 300
    n_max = 5
    geometry_encoder = ShapeGeometryEncoder(n_max=n_max, res=res, lam=1.0, encode_len=512)
    square_mask = get_square_mask(res=res)
    circle_mask = get_circle_mask(res=res)

    input_square_mask = MaskConvert.polar_mask_to_euclidean(square_mask, H=res, W=res, theta=None, rho=None, method="linear")

    square_encoded = geometry_encoder.encode_shape(square_mask.astype(np.float64))

    square_recon = geometry_encoder.reconstruct_shape(square_encoded)

    input_square_recon = MaskConvert.polar_mask_to_euclidean(square_recon, H=res*10, W=res*10, theta=None, rho=None, method="linear")

    input_square_mask = plot_utils.mask_to_rgb(input_square_mask, color_hex="#0078D4", bg_color=(1, 1, 1))
    input_square_recon = plot_utils.mask_to_rgb(input_square_recon, color_hex="#0078D4", bg_color=(1, 1, 1))

    plot_utils.plot_mask(input_square_mask, 
                         title='Original Square Mask (Euclidean)',
                         save_path="hello_original_square_mask_euclidean.png")
    
    plot_utils.plot_mask(input_square_recon, 
                         title='Reconstructed Square Mask (Euclidean)',
                         save_path="hello_reconstructed_square_mask_euclidean.png")

    mse_err = np.mean((square_mask.astype(np.float64) - square_recon)**2)
    print(f"Square reconstruction MSE error: {mse_err}")
    print(f"Square encoded length: {len(square_encoded)}")

    circle_encoded = geometry_encoder.encode_shape(circle_mask.astype(np.float64))
    print(f"Circle encoded length: {len(circle_encoded)}")
    print(f"Circle encoded coeffs (first 10): {circle_encoded[:10]}")

# if __name__ == "__main__":
#     main()