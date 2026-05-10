#!/usr/bin/env python3
"""Smoke tests for the XShapeEnc encoder suite.

Three self-contained tests are provided:

* **pose**          – :class:`ShapePoseEncoder.ShapePoseEncoder`: encodes a
  synthetic pose vector and measures the L2 reconstruction error.
* **geometry**      – :class:`ShapeGeometryEncoder.ShapeGeometryEncoder`:
  encodes canonical square / circle masks and reports reconstruction MSE.
* **geometry_pose** – :class:`ShapeGeometryPoseEncoder.ShapeGeometryPoseEncoder`:
  performs joint geometry + pose encoding of a square mask.

Usage
-----
Run all tests::

    python quick_test.py

Run a specific subset with debug output::

    python quick_test.py --tests pose geometry --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import numpy as np

import ShapeGeometryEncoder
import ShapeGeometryPoseEncoder
import ShapePoseEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------

_DEFAULT_POSE_VEC: List[float] = [0.2, 0.5, 0.7, 0.9, 0.4]


def get_circle_mask(res: int = 300) -> np.ndarray:
    """Return a boolean mask for a filled unit circle in polar-sampled space.

    The mask is evaluated over a ``(res × res)`` grid of ``(ρ, θ)`` samples
    converted to Cartesian coordinates, and is ``True`` wherever
    ``x² + y² ≤ 1``.

    Parameters
    ----------
    res:
        Number of samples along both the radial and angular axes.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(res, res)``.
    """
    rho = np.linspace(0, 1, res)
    theta = np.linspace(0, 2 * np.pi, res)
    r, t = np.meshgrid(rho, theta)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x**2 + y**2 <= 1.0


def get_square_mask(res: int = 300) -> np.ndarray:
    """Return a boolean mask for the largest axis-aligned square in the unit disk.

    The half-side length is ``1 / √2 ≈ 0.707`` so that all four corners of the
    square lie exactly on the unit circle.

    Parameters
    ----------
    res:
        Number of samples along both the radial and angular axes.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(res, res)``.
    """
    rho = np.linspace(0, 1, res)
    theta = np.linspace(0, 2 * np.pi, res)
    r, t = np.meshgrid(rho, theta)
    x = r * np.cos(t)
    y = r * np.sin(t)
    half_len = 1.0 / np.sqrt(2)
    return (np.abs(x) <= half_len) & (np.abs(y) <= half_len)


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------


def test_pose_encoder(
    res: int = 300,
    encode_len: int = 128,
    pose_vec: Optional[List[float]] = None,
    m_p: int = 17,
    seed_sigma: float = 0.1,
) -> None:
    """Verify that :class:`ShapePoseEncoder.ShapePoseEncoder` recovers a pose vector.

    Encodes *pose_vec* using Zernike-based pose encoding and logs the L2
    difference between the original and the recovered vector.

    Parameters
    ----------
    res:
        Grid resolution passed to the encoder.
    encode_len:
        Length of the Zernike coefficient vector.
    pose_vec:
        Pose vector to encode.  Defaults to ``[0.2, 0.5, 0.7, 0.9, 0.4]``.
    m_p:
        Angular frequency mode used for pose embedding.
    seed_sigma:
        Standard deviation of the seed Gaussian.
    """
    if pose_vec is None:
        pose_vec = _DEFAULT_POSE_VEC

    encoder = ShapePoseEncoder.ShapePoseEncoder(
        res=res,
        K=len(pose_vec),
        L=encode_len,
        m_p=m_p,
        seed_sigma=seed_sigma,
    )
    _, coeffs, _, p_recovered = encoder.encode(pose_vec)

    pose_arr = np.asarray(pose_vec)
    error = float(np.linalg.norm(pose_arr - p_recovered))

    logger.info("Original pose vector : %s", pose_vec)
    logger.info("Recovered pose vector: %s", list(p_recovered))
    logger.info("L2 recovery error    : %.6f", error)
    logger.debug("Zernike coefficients : %s", coeffs)


def test_geometry_encoder(
    n_max: int = 5,
    res: int = 300,
    lam: float = 0.6,
    encode_len: int = 512,
) -> None:
    """Verify that :class:`ShapeGeometryEncoder.ShapeGeometryEncoder` reconstructs canonical shapes.

    Encodes a square and a circle mask, decodes them, and reports the
    per-pixel reconstruction MSE for each shape.

    Parameters
    ----------
    n_max:
        Maximum Zernike polynomial order.
    res:
        Grid resolution.
    lam:
        Frequency-propagation regularisation parameter λ.
    encode_len:
        Target encoding vector length.
    """
    encoder = ShapeGeometryEncoder.ShapeGeometryEncoder(
        n_max=n_max, res=res, lam=lam, encode_len=encode_len
    )

    square_mask = get_square_mask(res=res).astype(np.float64)
    circle_mask = get_circle_mask(res=res).astype(np.float64)

    square_recon = encoder.decode(encoder.encode(square_mask))
    circle_recon = encoder.decode(encoder.encode(circle_mask))

    mse_square = float(np.mean((square_mask - square_recon) ** 2))
    mse_circle = float(np.mean((circle_mask - circle_recon) ** 2))

    logger.info("Square mask reconstruction MSE: %.6f", mse_square)
    logger.info("Circle mask reconstruction MSE: %.6f", mse_circle)


def test_geometry_pose_encoder(
    n_max: int = 20,
    res: int = 300,
    lam: float = 0.6,
    encode_len: int = 512,
    pose_vec: Optional[List[float]] = None,
    beta: float = 1.0,
) -> None:
    """Verify that :class:`ShapeGeometryPoseEncoder.ShapeGeometryPoseEncoder` produces a joint encoding.

    Encodes a square mask together with a synthetic pose vector and logs the
    length and (at DEBUG level) the contents of the resulting joint encoding
    vector.

    Parameters
    ----------
    n_max:
        Maximum Zernike polynomial order.
    res:
        Grid resolution.
    lam:
        Frequency-propagation regularisation parameter λ.
    encode_len:
        Target encoding vector length.
    pose_vec:
        Pose vector to encode.  Defaults to ``[0.2, 0.5, 0.7, 0.9, 0.4]``.
    beta:
        Geometry / pose balance parameter.  ``0`` = pose only,
        ``2`` = geometry only, ``1`` = equal weight.
    """
    if pose_vec is None:
        pose_vec = _DEFAULT_POSE_VEC

    encoder = ShapeGeometryPoseEncoder.ShapeGeometryPoseEncoder(
        n_max=n_max,
        res=res,
        lam=lam,
        encode_len=encode_len,
        pose_vec_len=len(pose_vec),
        beta=beta,
    )

    square_mask = get_square_mask(res=res).astype(np.float64)
    encoding = encoder.encode(square_mask, pose_vec=pose_vec, mask_in_euclidean=False)

    logger.info("Geometry-pose encoding length: %d", len(encoding))
    logger.debug("Geometry-pose encoding       : %s", encoding)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_TESTS: tuple[str, ...] = ("pose", "geometry", "geometry_pose")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke tests for the XShapeEnc encoder suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=_ALL_TESTS,
        default=list(_ALL_TESTS),
        metavar="TEST",
        help=(
            "One or more tests to run.  "
            f"Available choices: {{{', '.join(_ALL_TESTS)}}}."
        ),
    )
    parser.add_argument(
        "--res",
        type=int,
        default=300,
        help="Grid resolution (number of radial and angular samples).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging (includes full coefficient vectors).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the XShapeEnc smoke-test suite.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when *None*.

    Returns
    -------
    int
        ``0`` on success, ``1`` if any test raises an unexpected exception.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if "pose" in args.tests:
            logger.info("=== ShapePoseEncoder ===")
            test_pose_encoder(res=args.res)

        if "geometry" in args.tests:
            logger.info("=== ShapeGeometryEncoder ===")
            test_geometry_encoder(res=args.res)

        if "geometry_pose" in args.tests:
            logger.info("=== ShapeGeometryPoseEncoder ===")
            test_geometry_pose_encoder(res=args.res)

    except Exception:
        logger.exception("Test suite terminated with an unexpected error.")
        return 1

    logger.info("All selected tests completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())