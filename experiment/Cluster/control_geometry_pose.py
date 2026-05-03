"""control_geometry_pose.py

Geometry-pose encoding evaluation via t-SNE clustering.

Three encoding strategies are compared — XShapeEnc, Addition, and
Concatenation — across four spatial quadrants at varying beta/alpha weights.
Results are saved as PDF figures named ``<method>_beta_<value>.pdf``.

Usage::

    python control_geometry_pose.py [--methods XShapeEnc Addition Concate]
                                    [--xshapeenc-betas 0.2 1.0 1.8]
                                    [--baseline-alphas 0.01 0.04 0.06]
"""

import argparse
import copy
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import TSNE

sys.path.append('../../')
sys.path.append('../../utils/')

import ShapeGeometryEncoder
import ShapeGeometryPoseEncoder
import ShapePoseEncoder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHAPE_IDS: List[int] = [4, 15, 39, 49]
AREA_IDS: List[str] = ['topleft', 'topright', 'bottomleft', 'bottomright']

AREA_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    'bottomright': (227 / 255, 106 / 255, 107 / 255, 1.0),   # red
    'topleft':     (107 / 255, 188 / 255, 108 / 255, 1.0),   # green
    'topright':    ( 99 / 255, 158 / 255, 202 / 255, 1.0),   # blue
    'bottomleft':  (255 / 255, 166 / 255,  98 / 255, 1.0),   # orange
}

_SHAPE_INFO_PATH = Path('data') / 'shape_info.pkl'

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_shape_geometries() -> dict:
    """Load shape geometry information from disk.

    Returns:
        dict: Mapping from shape ID to shape metadata (``polar_mask``,
              ``cart_mask``, ``pose_vec``, …).

    Raises:
        FileNotFoundError: If ``data/shape_info.pkl`` is not found.
    """
    with open(_SHAPE_INFO_PATH, 'rb') as f:
        return pickle.load(f)

def get_shape_pose_vecs(area_id, add_scale, num = 4):
    area_pad = 10
    area_range = 100
    scale_range = [0.8, 1.2]

    topleft_x_range = [0+area_pad, area_range//2 - area_pad]
    topleft_y_range = [0+area_pad, area_range//2 - area_pad]

    topright_x_range = [area_range//2 + area_pad, area_range - area_pad]
    topright_y_range = [0+area_pad, area_range//2 - area_pad]

    bottomleft_x_range = [0+area_pad, area_range//2 - area_pad]
    bottomleft_y_range = [area_range//2 + area_pad, area_range - area_pad]

    bottomright_x_range = [area_range//2 + area_pad, area_range - area_pad]
    bottomright_y_range = [area_range//2 + area_pad, area_range - area_pad]

    if area_id == 'topleft':
        x_coords = np.random.uniform(topleft_x_range[0], topleft_x_range[1], num)
        y_coords = np.random.uniform(topleft_y_range[0], topleft_y_range[1], num)
    elif area_id == 'topright':
        x_coords = np.random.uniform(topright_x_range[0], topright_x_range[1], num)
        y_coords = np.random.uniform(topright_y_range[0], topright_y_range[1], num)
    elif area_id == 'bottomleft':
        x_coords = np.random.uniform(bottomleft_x_range[0], bottomleft_x_range[1], num)
        y_coords = np.random.uniform(bottomleft_y_range[0], bottomleft_y_range[1], num)
    elif area_id == 'bottomright':
        x_coords = np.random.uniform(bottomright_x_range[0], bottomright_x_range[1], num)
        y_coords = np.random.uniform(bottomright_y_range[0], bottomright_y_range[1], num)
    else:
        raise ValueError("area_id must be one of 'topleft', 'topright', 'bottomleft', 'bottomright'")

    scales = np.random.uniform(scale_range[0], scale_range[1], num)

    #final coord: [X, Y, Scale]
    if add_scale:
        coords = np.stack([x_coords/area_range, y_coords/area_range, scales], axis=1).squeeze()
    else:
        coords = np.stack([x_coords/area_range, y_coords/area_range], axis=1).squeeze()

    return coords

def get_shape_pose_vecs_bak(
    area_id: str,
    add_scale: bool,
    num: int = 4,
) -> np.ndarray:
    """Sample random pose vectors within the specified spatial quadrant.

    The canvas is divided into four equal quadrants over a [0, 100] grid.
    Sampled coordinates are normalised to [0, 1].

    Args:
        area_id: One of ``'topleft'``, ``'topright'``, ``'bottomleft'``,
                 ``'bottomright'``.
        add_scale: Append a scale value sampled from ``[0.8, 1.2]`` when
                   ``True``.
        num: Number of pose vectors to sample.

    Returns:
        np.ndarray: Array of shape ``(num, D)`` where *D* is 2 or 3 depending
                    on *add_scale*.  Squeezed to ``(D,)`` when *num* is 1.

    Raises:
        ValueError: If *area_id* is not recognised.
    """
    area_pad = 10
    area_range = 100
    scale_range = (0.8, 1.2)
    half = area_range // 2

    x_bounds: Dict[str, Tuple[int, int]] = {
        'topleft':     (area_pad,        half - area_pad),
        'topright':    (half + area_pad, area_range - area_pad),
        'bottomleft':  (area_pad,        half - area_pad),
        'bottomright': (half + area_pad, area_range - area_pad),
    }
    y_bounds: Dict[str, Tuple[int, int]] = {
        'topleft':     (area_pad,        half - area_pad),
        'topright':    (area_pad,        half - area_pad),
        'bottomleft':  (half + area_pad, area_range - area_pad),
        'bottomright': (half + area_pad, area_range - area_pad),
    }

    if area_id not in x_bounds:
        raise ValueError(
            f"area_id must be one of {sorted(x_bounds)}, got '{area_id}'."
        )

    x_coords = np.random.uniform(*x_bounds[area_id], num) / area_range
    y_coords = np.random.uniform(*y_bounds[area_id], num) / area_range
    # Always sample scales to preserve the RNG state regardless of add_scale,
    # ensuring reproducible results across consecutive calls.
    scales = np.random.uniform(*scale_range, num)

    if add_scale:
        coords = np.stack([x_coords, y_coords, scales], axis=1)
    else:
        coords = np.stack([x_coords, y_coords], axis=1)

    return coords.squeeze() if num == 1 else coords

def get_spatially_grounded_shape(
    shape_info: dict,
    add_scale: bool = True,
) -> Tuple[dict, dict, dict, dict]:
    """Create four quadrant-specific copies of *shape_info* with sampled poses.

    Args:
        shape_info: Shape metadata dict as returned by
                    :func:`get_shape_geometries`.
        add_scale: Whether to include a scale dimension in the pose vector.

    Returns:
        Tuple ``(topleft, topright, bottomleft, bottomright)`` — four deep
        copies of *shape_info* with updated ``'pose_vec'`` fields.
    """
    area_shapes = {area_id: copy.deepcopy(shape_info) for area_id in AREA_IDS}
    for shape_id in SHAPE_IDS:
        for area_id in AREA_IDS:
            area_shapes[area_id][shape_id]['pose_vec'] = get_shape_pose_vecs(
                area_id, add_scale=add_scale, num=1
            )
    return (
        area_shapes['topleft'],
        area_shapes['topright'],
        area_shapes['bottomleft'],
        area_shapes['bottomright'],
    )


def _build_sample_lists(
    area_shapes: Dict[str, dict],
) -> Tuple[List, List, List, List[str], List[int]]:
    """Flatten per-area shape dicts into aligned sample lists.

    Args:
        area_shapes: Mapping from area name to per-shape metadata dicts.

    Returns:
        Tuple of five lists:
        ``(geometry_masks, pose_vecs, cart_masks, area_ids, shape_ids)``.
    """
    geometry_masks: List = []
    pose_vecs: List = []
    cart_masks: List = []
    area_ids: List[str] = []
    shape_ids: List[int] = []
    for shape_id in SHAPE_IDS:
        for area_id in AREA_IDS:
            s = area_shapes[area_id][shape_id]
            geometry_masks.append(s['polar_mask'])
            pose_vecs.append(s['pose_vec'])
            cart_masks.append(s['cart_mask'])
            area_ids.append(area_id)
            shape_ids.append(shape_id)

    return geometry_masks, pose_vecs, cart_masks, area_ids, shape_ids

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
    """
    Run t-SNE clustering and visualize with original cartesian shape marks.
    
    Args:
        feat_list: List of encoded features
        cart_masks: List of cartesian masks for each feature
        area_ids: List of area IDs ('topleft', 'topright', etc.) for each feature
        shape_ids: List of shape IDs for each feature
        beta: Beta parameter for filename
        method_name: Name of the method for filename
    """
    feat_array = np.array(feat_list)
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate=100)
    feat_2d = tsne.fit_transform(feat_array)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    
    # Fixed zoom factor for consistent shape size across all visualizations
    zoom_factor = 0.15
    
    for i in range(len(feat_list)):
        area_id = area_ids[i]
        shape_id = shape_ids[i]
        cart_mask = cart_masks[i]
        color = AREA_COLORS[area_id]  # Already RGBA tuple with transparency
        mask_rgba = np.zeros((*cart_mask.shape, 4))
        mask_rgba[cart_mask > 0.5] = color
        mask_rgba[cart_mask <= 0.5] = [1, 1, 1, 0]  # Transparent background
        
        # Create an OffsetImage with the shape
        imagebox = OffsetImage(mask_rgba, zoom=zoom_factor)
        imagebox.image.axes = ax
        
        # Place the shape at the t-SNE coordinates
        ab = AnnotationBbox(imagebox, (feat_2d[i, 0], feat_2d[i, 1]),
                           frameon=False, pad=0)
        ax.add_artist(ab)
    
    # Calculate axis limits
    x_range = feat_2d[:, 0].max() - feat_2d[:, 0].min()
    y_range = feat_2d[:, 1].max() - feat_2d[:, 1].min()
    
    # Remove axis labels and ticks but keep grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)  # Remove tick marks
    
    # Remove boundary box (spines) but keep grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Adjust axis limits to give some padding
    x_pad = x_range * 0.15
    y_pad = y_range * 0.15
    ax.set_xlim(feat_2d[:, 0].min() - x_pad, feat_2d[:, 0].max() + x_pad)
    ax.set_ylim(feat_2d[:, 1].min() - y_pad, feat_2d[:, 1].max() + y_pad)
    
    plt.tight_layout()
    # Save as PDF with high DPI but optimized file size
    plt.savefig(f'{method_name}_beta_{beta}.pdf', dpi=300, 
                bbox_inches='tight', pad_inches=0.1, 
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved visualization to {method_name}_beta_{beta}.pdf")
    plt.close(fig)

def run_tSNE_clustering(
    feat_list: List,
    cart_masks: List[np.ndarray],
    area_ids: List[str],
    beta: float,
    method_name: str = 'XShapeEnc',
    _save: bool = True,
) -> None:
    """Embed *feat_list* with t-SNE and render a glyph scatter plot.

    Each sample is drawn as its Cartesian shape mask, coloured by quadrant.
    The figure is saved as ``<method_name>_beta_<beta>.pdf`` when *_save* is
    ``True``.  t-SNE always runs so that numpy global RNG consumption is
    identical regardless of *_save*.

    Args:
        feat_list: List of 1-D feature vectors.
        cart_masks: Binary Cartesian masks used as visual glyphs.
        area_ids: Quadrant label for each sample.
        beta: Used only to construct the output filename.
        method_name: Prefix for the output filename.
        _save: If ``False``, skip figure creation; t-SNE still runs.
    """
    feat_array = np.array(feat_list)
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate=100)
    feat_2d = tsne.fit_transform(feat_array)

    if not _save:
        return

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    zoom_factor = 0.15

    for i, (area_id, cart_mask) in enumerate(zip(area_ids, cart_masks)):
        color = AREA_COLORS[area_id]
        mask_rgba = np.zeros((*cart_mask.shape, 4))
        mask_rgba[cart_mask > 0.5] = color
        mask_rgba[cart_mask <= 0.5] = [1, 1, 1, 0]
        imagebox = OffsetImage(mask_rgba, zoom=zoom_factor)
        imagebox.image.axes = ax
        ab = AnnotationBbox(
            imagebox, (feat_2d[i, 0], feat_2d[i, 1]), frameon=False, pad=0
        )
        ax.add_artist(ab)

    x_range = feat_2d[:, 0].max() - feat_2d[:, 0].min()
    y_range = feat_2d[:, 1].max() - feat_2d[:, 1].min()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(
        feat_2d[:, 0].min() - x_range * 0.15,
        feat_2d[:, 0].max() + x_range * 0.15,
    )
    ax.set_ylim(
        feat_2d[:, 1].min() - y_range * 0.15,
        feat_2d[:, 1].max() + y_range * 0.15,
    )
    plt.tight_layout()
    out_path = f'{method_name}_beta_{beta}.pdf'
    plt.savefig(
        out_path, dpi=300, bbox_inches='tight', pad_inches=0.1,
        facecolor=fig.get_facecolor(), edgecolor='none',
    )
    plt.close(fig)
    logger.info('Saved visualisation to %s', out_path)

# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def run_addition_cluster(beta: float = 0.5, _save: bool = True) -> None:
    """Run the Addition-fusion clustering experiment.

    Geometry and pose encodings are each produced at length *encode_len* and
    fused via ``geo + beta * pose``.

    Args:
        beta: Scalar weight applied to the pose encoding before addition.
        _save: Whether to produce and save the output figure.
    """
    lam = 0.2
    encode_len = 512
    shape_info = get_shape_geometries()
    topleft_shape, topright_shape, bottomleft_shape, bottomright_shape = \
        get_spatially_grounded_shape(shape_info, add_scale=False)

    area_shapes = {
        'topleft': topleft_shape,
        'topright': topright_shape,
        'bottomleft': bottomleft_shape,
        'bottomright': bottomright_shape,
    }
    shape_geo_encoder = ShapeGeometryEncoder.ShapeGeometryEncoder(
        encode_len=encode_len, res=300, lam=lam
    )
    shape_pose_encoder = ShapePoseEncoder.ShapePoseEncoder(
        res=300, K=2, L=512, m_p=10, seed_sigma=0.1
    )

    geometry_masks, pose_vecs, cart_masks, area_id_list, shape_id_list = \
        _build_sample_lists(area_shapes)

    feat_list = []
    for shape_mask, pose_vec in zip(geometry_masks, pose_vecs):
        geo_enc = np.array(
            shape_geo_encoder.encode(shape_mask.astype(np.float64)), dtype=np.float32
        )
        pose_enc = np.array(
            shape_pose_encoder.encode(pose_vec.astype(np.float64), just_return_coeff=True),
            dtype=np.float32,
        )
        feat_list.append((geo_enc + beta * pose_enc).tolist())

    run_tSNE_clustering(
        feat_list, cart_masks, area_id_list,
        beta=beta, method_name='Addition', _save=_save,
    )

def run_concate_cluster(beta: float = 0.5, _save: bool = True) -> None:
    """Run the Concatenation-fusion clustering experiment.

    Half the encoding budget goes to geometry and half to pose.
    The pose half is scaled by *beta* before concatenation.

    Args:
        beta: Scalar weight applied to the pose encoding before concatenation.
        _save: Whether to produce and save the output figure.

    Raises:
        ValueError: If the combined encoding length is unexpected.
    """
    lam = 0.2
    encode_len = 256
    pose_vec_len = 2
    shape_info = get_shape_geometries()
    topleft_shape, topright_shape, bottomleft_shape, bottomright_shape = \
        get_spatially_grounded_shape(shape_info, add_scale=False)

    area_shapes = {
        'topleft': topleft_shape,
        'topright': topright_shape,
        'bottomleft': bottomleft_shape,
        'bottomright': bottomright_shape,
    }
    shape_geo_encoder = ShapeGeometryEncoder.ShapeGeometryEncoder(
        encode_len=encode_len, res=300, lam=lam
    )
    shape_pose_encoder = ShapePoseEncoder.ShapePoseEncoder(
        res=300, K=pose_vec_len, L=encode_len, m_p=10, seed_sigma=0.1
    )

    geometry_masks, pose_vecs, cart_masks, area_id_list, shape_id_list = \
        _build_sample_lists(area_shapes)

    feat_list = []
    for shape_mask, pose_vec in zip(geometry_masks, pose_vecs):
        geo_enc = shape_geo_encoder.encode(shape_mask.astype(np.float64))[:encode_len]
        pose_enc = (
            beta
            * np.array(
                shape_pose_encoder.encode(
                    pose_vec.astype(np.float64), just_return_coeff=True
                ),
                dtype=np.float32,
            )
        ).tolist()
        combined = geo_enc + pose_enc
        if len(combined) != 2 * encode_len:
            raise ValueError(
                f'Combined encoding length {len(combined)} does not match '
                f'expected {2 * encode_len}.'
            )
        feat_list.append(combined)

    run_tSNE_clustering(
        feat_list, cart_masks, area_id_list,
        beta=beta, method_name='Concate', _save=_save,
    )


def run_XShapeEnc_cluster(beta: float = 0.5, _save: bool = True) -> None:
    """Run the XShapeEnc joint geometry-pose encoding clustering experiment.

    A single :class:`ShapeGeometryPoseEncoder` produces a unified encoding
    that jointly weighs geometry and pose information via *beta*.

    Args:
        beta: Weighting parameter forwarded to
              :class:`ShapeGeometryPoseEncoder.ShapeGeometryPoseEncoder`.
        _save: Whether to produce and save the output figure.
    """
    lam = 0.2
    encode_len = 512
    pose_vec_len = 2
    shape_info = get_shape_geometries()
    topleft_shape, topright_shape, bottomleft_shape, bottomright_shape = \
        get_spatially_grounded_shape(shape_info, add_scale=False)

    area_shapes = {
        'topleft': topleft_shape,
        'topright': topright_shape,
        'bottomleft': bottomleft_shape,
        'bottomright': bottomright_shape,
    }

    geometry_masks, pose_vecs, cart_masks, area_id_list, shape_id_list = \
        _build_sample_lists(area_shapes)

    geo_pose_encoder = ShapeGeometryPoseEncoder.ShapeGeometryPoseEncoder(
        n_max=20,
        res=300,
        lam=lam,
        encode_len=encode_len,
        pose_vec_len=pose_vec_len,
        seed_sigma=0.1,
        beta=beta,
    )

    feat_list = [
        geo_pose_encoder.encode(
            shape_mask.astype(np.float64),
            pose_vec.astype(np.float64),
            run_freqprop=True,
            mask_in_euclidean=False,
        )
        for shape_mask, pose_vec in zip(geometry_masks, pose_vecs)
    ]

    run_tSNE_clustering(
        feat_list, cart_masks, area_id_list,
        beta=beta, method_name='XShapeEnc', _save=_save,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run geometry-pose t-SNE clustering experiments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['XShapeEnc', 'Addition', 'Concate', 'all'],
        default=['all'],
        help='Which method(s) to evaluate.',
    )
    parser.add_argument(
        '--xshapeenc-betas',
        nargs='+',
        type=float,
        default=[0.2, 1.0, 1.8],
        metavar='BETA',
        help='Beta values for the XShapeEnc experiment.',
    )
    parser.add_argument(
        '--baseline-alphas',
        nargs='+',
        type=float,
        default=[0.01, 0.04, 0.06],
        metavar='ALPHA',
        help='Alpha (beta) values for the Addition and Concatenation experiments.',
    )
    return parser

def main() -> None:
    """Parse arguments and dispatch the requested clustering experiments.
    Methods are always executed in the original canonical order — XShapeEnc
    betas first, then Addition/Concate interleaved per alpha — to guarantee
    that numpy's global RNG is consumed identically regardless of which
    methods are selected for output.  Passing ``--methods Addition`` will
    therefore produce figures identical to those from the original full run.
    """
    args = _build_parser().parse_args()
    run_all = 'all' in args.methods
    save_xshapeenc = run_all or 'XShapeEnc' in args.methods
    save_addition  = run_all or 'Addition'  in args.methods
    save_concate   = run_all or 'Concate'   in args.methods

    # Phase 1: XShapeEnc — always runs to consume the same RNG draws as the
    for beta in args.xshapeenc_betas:
        if save_xshapeenc:
            logger.info('XShapeEnc — beta=%.3f', beta)
        run_XShapeEnc_cluster(beta=beta, _save=save_xshapeenc)

    # Phase 2: Addition and Concate
    for alpha in args.baseline_alphas:
        if save_addition:
            logger.info('Addition  — alpha=%.3f', alpha)
        run_addition_cluster(beta=alpha, _save=save_addition)
        if save_concate:
            logger.info('Concate   — alpha=%.3f', alpha)
        run_concate_cluster(beta=alpha, _save=save_concate)

if __name__ == '__main__':
    main()