"""
ShapeGeometryCluster.py
=======================
Evaluate shape-encoding methods by visualising their t-SNE clustering on
augmented binary masks.

For each method a PDF figure is saved to ``--output-dir``.  Every shape
class is plotted in a distinct colour so that intra-class compactness and
inter-class separation are immediately visible.

Usage
-----
    python ShapeGeometryCluster.py [options]

    --mask-file   Path to the pickle file containing shape masks.
                  (default: data/shape_geometry_masks.pkl)
    --output-dir  Directory where output PDFs are saved.
                  (default: output/)
    --encode-len  Length of the encoding vector for each method.
                  (default: 512)
    --aug-num     Number of augmented copies per original mask.
                  (default: 200)
    --seed        Random seed for reproducibility.
                  (default: 42)
    --perplexity  t-SNE perplexity.
                  (default: 30)
    --methods     Comma-separated list of methods to run.
                  Available: XShapeEnc, PointSet, AngularSweep, AngularSweepPE,
                             DistanceTransform, ShapeDist, EFT, Shape2Vec,
                             ResNet18, ViT, PointSetPE
                  (default: all)
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../../'))
sys.path.insert(0, os.path.join(_HERE, '../../utils/'))

from ShapeGeometryEncoder import ShapeGeometryEncoder  # noqa: E402
import MaskAugmentor   # noqa: E402
import MaskConvert     # noqa: E402
import Baselines       # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------

def get_XShapeEnc_encodings(cart_masks, encode_len=512):
    n_max, res = 100, 300
    encoder = ShapeGeometryEncoder(n_max=n_max, res=res, lam=0.6, encode_len=encode_len)
    encods = []
    for mask in cart_masks:
        polar = MaskConvert.euclidean_mask_to_polar(
            mask,
            theta_resolution=mask.shape[0],
            rho_resolution=mask.shape[1],
        )[2]
        encods.append(encoder.encode(polar.astype(np.float64), return_raw_coeffs=False))
    return np.array(encods, dtype=np.float64)


def get_PointSet_encodings(cart_masks, encode_len=512):
    encods = [
        Baselines.Baseline.regular_grid_point_encoding(m, encode_len=encode_len)
        for m in cart_masks
    ]
    return np.array(encods, dtype=np.float64)


def get_angular_sweep_encodings(cart_masks, encode_len=512):
    encods = [
        Baselines.Baseline.angular_sweep_encoding(m, encode_len=encode_len)
        for m in cart_masks
    ]
    return np.array(encods, dtype=np.float64)


def get_angular_sweep_pe_encodings(cart_masks, encode_len=512):
    encods = []
    for mask in cart_masks:
        boundary = Baselines.Baseline.angular_sweep_encoding(mask, encode_len=encode_len)
        enc = Baselines.Baseline.sinusoidal_encoding_1d(
            np.array(boundary, dtype=np.float32), encode_len=encode_len
        )
        encods.append(enc)
    return np.array(encods, dtype=np.float64)


def get_distance_transform_encodings(cart_masks, encode_len=512):
    encods = [
        Baselines.Baseline.distance_transform_encoding(m, encode_len=encode_len)
        for m in cart_masks
    ]
    return np.array(encods, dtype=np.float64)


def get_shape_dist_encodings(cart_masks, encode_len=512):
    encods = [
        Baselines.Baseline.shape_distribution_D2_boundary_deterministic(m, encode_len=encode_len)
        for m in cart_masks
    ]
    return np.array(encods, dtype=np.float64)


def get_eft_encodings(cart_masks, encode_len=512):
    encods = []
    for mask in cart_masks:
        enc = Baselines.Baseline.eft_encode_shape(mask, encode_len=encode_len)
        if len(enc) != encode_len:
            raise ValueError(
                f"EFT returned {len(enc)} values but expected {encode_len}."
            )
        encods.append(enc)
    return np.array(encods, dtype=np.float64)


def get_shape2vec_encodings(cart_masks, encode_len=512):
    encods = [
        Baselines.Baseline.space2vec_shape_encoding(m, encode_len=encode_len)
        for m in cart_masks
    ]
    return np.array(encods, dtype=np.float64)


def get_resnet18_encodings(cart_masks, encode_len=512, layer='avgpool'):
    encods = [
        Baselines.Baseline.resnet18_shape_encoding(m, encode_len=encode_len, layer=layer)
        for m in cart_masks
    ]
    return np.array(encods, dtype=np.float64)


def get_vit_encodings(cart_masks, encode_len=512):
    encods = [
        Baselines.Baseline.vit_shape_encoding(m, encode_len=encode_len)
        for m in cart_masks
    ]
    return np.array(encods, dtype=np.float64)


def get_pointset_pe_encodings(cart_masks, encode_len=512):
    encods = []
    for mask in cart_masks:
        pts = Baselines.Baseline.regular_grid_point_encoding_coord(
            mask, encode_len=encode_len, flatten=False
        )
        enc = Baselines.Baseline.sinusoidal_encoding(
            np.array(pts, dtype=np.float32), encode_len=encode_len
        )
        encods.append(enc)
    return np.array(encods, dtype=np.float64)


# ---------------------------------------------------------------------------
# t-SNE + plotting
# ---------------------------------------------------------------------------

def run_tsne_and_plot(encods, aug_num, method_name, output_dir, perplexity=30):
    """Fit t-SNE on *encods* and save a scatter-plot PDF to *output_dir*."""
    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        perplexity=perplexity,
        random_state=0,
    )
    Z = tsne.fit_transform(encods)

    n_shapes = 4
    shape_names = [f'shape{i}' for i in range(n_shapes)]
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots()
    for label in range(n_shapes):
        idx = list(range(label * (aug_num + 1), (label + 1) * (aug_num + 1)))
        ax.scatter(
            Z[idx, 0], Z[idx, 1],
            s=8,
            color=colours[label % len(colours)],
            label=shape_names[label],
        )

    ax.axis('off')

    out_path = os.path.join(output_dir, f'{method_name}.pdf')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def build_method_registry(cart_masks, encode_len):
    """Return an ordered dict mapping method name -> encoding callable."""
    return {
        'XShapeEnc':       lambda: get_XShapeEnc_encodings(cart_masks, encode_len),
        'PointSet':        lambda: get_PointSet_encodings(cart_masks, encode_len),
        'AngularSweep':    lambda: get_angular_sweep_encodings(cart_masks, encode_len),
        'AngularSweepPE':  lambda: get_angular_sweep_pe_encodings(cart_masks, encode_len),
        'DistanceTransform': lambda: get_distance_transform_encodings(cart_masks, encode_len),
        'ShapeDist':       lambda: get_shape_dist_encodings(cart_masks, encode_len),
        'EFT':             lambda: get_eft_encodings(cart_masks, encode_len),
        'Shape2Vec':       lambda: get_shape2vec_encodings(cart_masks, encode_len),
        'ResNet18':        lambda: get_resnet18_encodings(cart_masks, encode_len, layer='avgpool'),
        'ViT':             lambda: get_vit_encodings(cart_masks, encode_len),
        'PointSetPE':      lambda: get_pointset_pe_encodings(cart_masks, encode_len),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='t-SNE clustering visualisation for shape encoding methods.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mask-file', default='data/shape_geometry_masks.pkl',
                        help='Pickle file containing the original shape masks.')
    parser.add_argument('--output-dir', default='output',
                        help='Directory where output PDFs are saved.')
    parser.add_argument('--encode-len', type=int, default=512,
                        help='Encoding vector length.')
    parser.add_argument('--aug-num', type=int, default=200,
                        help='Number of augmented copies per original mask.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity.')
    parser.add_argument('--methods', default='all',
                        help='Comma-separated list of methods to run, or "all".')
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load original masks
    # ------------------------------------------------------------------
    log.info('Loading masks from %s', args.mask_file)
    with open(args.mask_file, 'rb') as f:
        loaded = pickle.load(f)
    ori_masks = loaded['cart_masks']
    log.info('Loaded %d original masks.', len(ori_masks))

    # ------------------------------------------------------------------
    # Step 2: Augment masks
    # ------------------------------------------------------------------
    augmentor = MaskAugmentor.MaskAugmentor(
        rotation_range=(-np.pi / 3, np.pi / 3),
        scale_range=(0.8, 1.1),
        shear_range=(-0.15, 0.15),
        elastic_sigma=50.1,
        elastic_alpha=50.1,
        fit_mode='disk',
        margin=0.90,
    )
    rng = np.random.default_rng(args.seed)

    cart_masks = []
    for ori_mask in ori_masks:
        cart_masks.append(ori_mask)
        for _ in range(args.aug_num):
            cart_masks.append(augmentor.augment(ori_mask, rng=rng))

    log.info('Total masks after augmentation: %d', len(cart_masks))

    # ------------------------------------------------------------------
    # Step 3: Build method registry and select requested methods
    # ------------------------------------------------------------------
    registry = build_method_registry(cart_masks, args.encode_len)

    if args.methods.lower() == 'all':
        selected = list(registry.keys())
    else:
        selected = [m.strip() for m in args.methods.split(',')]
        unknown = [m for m in selected if m not in registry]
        if unknown:
            log.error('Unknown method(s): %s. Available: %s',
                      unknown, list(registry.keys()))
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Encode → t-SNE → save figure  (per method)
    # ------------------------------------------------------------------
    failed = []
    for method_name in selected:
        log.info('[%s] computing encodings ...', method_name)
        try:
            encods = registry[method_name]()
        except Exception as exc:
            log.error('[%s] encoding failed: %s', method_name, exc)
            failed.append(method_name)
            continue

        log.info('[%s] running t-SNE and plotting ...', method_name)
        try:
            out_path = run_tsne_and_plot(
                encods,
                aug_num=args.aug_num,
                method_name=method_name,
                output_dir=args.output_dir,
                perplexity=args.perplexity,
            )
            log.info('[%s] saved -> %s', method_name, out_path)
        except Exception as exc:
            log.error('[%s] plotting failed: %s', method_name, exc)
            failed.append(method_name)

    if failed:
        log.warning('The following methods encountered errors: %s', failed)
    else:
        log.info('All methods completed successfully.')


if __name__ == '__main__':
    main()
