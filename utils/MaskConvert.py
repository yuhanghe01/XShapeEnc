import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

def remove_small_disconnected_shapes(
    mask: np.ndarray,
    area_ratio: float = 0.10,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Remove small, disconnected foreground blobs from a binary mask.

    Rules:
      • Keep the 'main shape' = largest connected component (by area).
      • For all *other* components: remove if area < area_ratio * (H*W).
      • Leave any disconnected but large components intact.

    Args:
        mask: 2D array; foreground is 1 (or >0) and background is 0.
        area_ratio: Threshold as a fraction of total image area.
        connectivity: 1 for 4-connectivity, 2 for 8-connectivity.

    Returns:
        A binary mask (dtype like input) with small disconnected blobs removed.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    # Ensure boolean foreground
    fg = mask.astype(bool)

    if not fg.any():
        return np.zeros_like(mask)

    # Choose connectivity structure (4- or 8-connected)
    if connectivity == 1:
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]], dtype=np.uint8)
    else:
        structure = np.ones((3,3), dtype=np.uint8)

    # Label connected components
    labeled, num = ndimage.label(fg, structure=structure)
    if num == 0:
        return np.zeros_like(mask)

    # Component areas (bin 0 is background)
    areas = np.bincount(labeled.ravel())
    areas[0] = 0

    # Main shape = largest component
    main_label = areas.argmax()

    # Absolute pixel threshold
    H, W = mask.shape
    min_area = int(area_ratio * H * W)

    # Keep rule:
    #   - always keep main_label
    #   - also keep any component with area >= min_area
    keep_labels = set(np.where(areas >= min_area)[0].tolist() + [int(main_label)])

    # Build output mask
    keep = np.isin(labeled, list(keep_labels))
    return keep.astype(mask.dtype)


def _cartesian_grid(H, W):
    """
    Build a Cartesian coordinate grid spanning x,y in [-1, 1]×[-1, 1].
    Note: row index 0 is the TOP, so y decreases with row.
    """
    xs = np.linspace(-1.0, 1.0, W)
    ys = np.linspace(1.0, -1.0, H)   # top -> bottom
    X, Y = np.meshgrid(xs, ys)        # shape (H, W)

    return X, Y

def _polar_grid(theta_resolution, rho_resolution):
    """
    Build a regular polar grid: theta in [0, 2pi), rho in [0, 1].
    Returns (Theta, Rho) with shape (n_theta, n_rho).
    """
    theta = np.linspace(0.0, 2*np.pi, theta_resolution, endpoint=False)
    rho   = np.linspace(0.0, 1.0, rho_resolution, endpoint=True)
    R, T  = np.meshgrid(rho, theta)  # (n_theta, n_rho) order

    return T, R, theta, rho

def euclidean_mask_to_polar(mask_euclid, theta_resolution=512, rho_resolution=512, method="linear"):
    """
    Convert a Euclidean 2D mask (H×W) defined over the square [-1,1]^2
    into a polar mask on a (theta × rho) grid over the unit disk.

    Parameters
    ----------
    mask_euclid : (H, W) array-like (bool or 0/1/float)
        Assumed to cover x,y in [-1,1] with the unit disk centered in the image.
        Pixels outside the unit disk can be anything; they won't affect sampling inside.
    theta_resolution : int
        Number of angular samples (rows) in the polar grid.
    rho_resolution : int
        Number of radial samples (cols) in the polar grid.
    method : {"nearest", "linear"}
        Sampling method. "linear" uses SciPy if available; otherwise falls back to "nearest".

    Returns
    -------
    theta, rho : 1D arrays defining the polar grid
    mask_polar : (n_theta, n_rho) float array in [0,1]
    """
    mask_euclid = np.asarray(mask_euclid).astype(float)
    H, W = mask_euclid.shape

    # Build polar grid
    T, R, theta, rho = _polar_grid(theta_resolution, rho_resolution)

    # Polar -> Cartesian query points
    Xq = R * np.cos(T)   # in [-1,1]
    Yq = R * np.sin(T)   # in [-1,1]

    # Map (x,y) to (row, col) indices in the Euclidean image
    # x: -1..1  -> col: 0..W-1 ; y: 1..-1 -> row: 0..H-1
    col = (Xq + 1.0) * 0.5 * (W - 1)
    row = (1.0 - (Yq + 1.0) * 0.5) * (H - 1)

    if method == "linear":
        interp = RegularGridInterpolator(
            (np.arange(H), np.arange(W)), mask_euclid, method="linear",
            bounds_error=False, fill_value=0.0
        )
        pts = np.stack([row.ravel(), col.ravel()], axis=-1)
        Z = interp(pts).reshape(theta_resolution, rho_resolution)
    else:
        # Nearest neighbor fallback
        r = np.rint(row).astype(int)
        c = np.rint(col).astype(int)
        r = np.clip(r, 0, H-1)
        c = np.clip(c, 0, W-1)
        Z = mask_euclid[r, c].astype(float)

    # Outside unit disk -> 0 (R>1 won't happen, but guard anyway)
    Z[R > 1.0] = 0.0

    return theta, rho, Z

# -------------------------------
# 2) Polar -> Euclidean
# -------------------------------

def polar_mask_to_euclidean(mask_polar, H=1024, W=1024, theta=None, rho=None, method="linear"):
    """
    Convert a polar mask (theta × rho over unit disk) to a Euclidean 2D mask (H×W)
    defined over the square [-1,1]^2 with the unit disk centered.

    Parameters
    ----------
    mask_polar : (n_theta, n_rho) array-like (bool/float)
        Values defined on a regular (theta, rho) grid, theta in [0, 2π), rho in [0,1].
    H, W : int
        Output Euclidean resolution.
    theta, rho : 1D arrays or None
        If None, they are inferred as regular grids spanning [0,2π) and [0,1].
    method : {"nearest", "linear"}
        Sampling method. "linear" uses SciPy if available; otherwise falls back to "nearest".

    Returns
    -------
    mask_euclid : (H, W) float array in [0,1]
    """
    Zp = np.asarray(mask_polar).astype(float)
    n_theta, n_rho = Zp.shape

    # Define theta, rho axes if not supplied
    if theta is None:
        theta = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    if rho is None:
        rho   = np.linspace(0.0, 1.0,     n_rho,   endpoint=True)

    # Build Euclidean target grid
    X, Y = _cartesian_grid(H, W)       # in [-1,1]^2
    R    = np.sqrt(X**2 + Y**2)
    T    = np.mod(np.arctan2(Y, X), 2*np.pi)
    inside = R <= 1.0

    # Handle angular wrap-around by extending theta and duplicating first row
    theta_ext = np.concatenate([theta, [theta[0] + 2*np.pi]])
    Zp_ext    = np.vstack([Zp, Zp[0:1, :]])

    if method == "linear":
        interp = RegularGridInterpolator(
            (theta_ext, rho),
            Zp_ext,
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
        pts = np.stack([T.ravel(), R.ravel()], axis=-1)
        Ze = interp(pts).reshape(H, W)
    else:
        # Nearest neighbor fallback in (theta, rho)
        t_idx = np.searchsorted(theta, T, side="right") - 1
        t_idx = np.mod(t_idx, n_theta)  # wrap
        r_idx = np.searchsorted(rho, R, side="right") - 1
        r_idx = np.clip(r_idx, 0, n_rho-1)
        Ze = Zp[t_idx, r_idx].astype(float)

    Ze[~inside] = 0.0

    Ze[Ze>0.2] = 1.0
    Ze[Ze<=0.2] = 0.0


    Ze = remove_small_disconnected_shapes(Ze, area_ratio=0.1, connectivity=2)

    return Ze

# if __name__ == "__main__":
#     # Build a Euclidean square mask inscribed in the unit circle
#     H, W = 400, 400
#     X, Y = _cartesian_grid(H, W)
#     half = 1/np.sqrt(2)
#     mask_e = (np.abs(X) <= half) & (np.abs(Y) <= half)

#     # Euclidean -> Polar
#     theta, rho, mask_p = euclidean_mask_to_polar(mask_e, theta_resolution=720, rho_resolution=512, method="linear")

#     # Polar -> Euclidean (back)
#     mask_e_rec = polar_mask_to_euclidean(mask_p, H=400, W=400, theta=theta, rho=rho, method="linear")


#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].imshow(mask_e, cmap="gray", origin="lower")
#     axs[0].set_title("Original Euclidean Mask")
#     axs[1].imshow(mask_p, cmap="gray", origin="lower", extent=[0,1,0,2*np.pi])
#     axs[1].set_title("Polar Mask")
#     axs[2].imshow(mask_e_rec, cmap="gray", origin="lower")
#     axs[2].set_title("Reconstructed Euclidean Mask")
#     plt.savefig("mask_conversion_example.png", bbox_inches='tight')

#     # (Optional) binarize for a strict mask view:
#     # mask_e_rec_bin = (mask_e_rec > 0.5)