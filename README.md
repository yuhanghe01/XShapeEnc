<p align="center">
<a href="https://arxiv.org/pdf/2604.07522v1"><img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper"></a>
<a href="https://yuhanghe01.github.io/XShapeEnc-Proj/"><img src="https://img.shields.io/badge/website-visit-blue?logo=github" alt="Project Website"></a>
<img src="https://img.shields.io/github/forks/yuhanghe01/XShapeEnc" alt="GitHub forks">
<img src="https://img.shields.io/github/stars/yuhanghe01/XShapeEnc" alt="GitHub stars">
</p>

## Training-free Spatially Grounded Geometric Shape Encoding (Technical Report)

[Yuhang He](https://yuhanghe01.github.io/)<br>
Microsoft Research

### Quick Test

1. To build up the environment `pip install requirements.txt`.

2. Run `quick_test.py` to experience shape geometry encoding, shape pose encoding and their joint encoding.

    ```python
    #Run all tests::
    python quick_test.py
    #Run a specific subset with debug output::
    python quick_test.py --tests pose geometry --verbose
    ```

3. XShapeCorpus dataset generation, go to [README.md](./XShapeCorpus/README.md).

### XShapeEnc Summary

 | Method | arbitrary shape? | high frequency? | training-free? | task-agnostic? | spatial-context? |
|---|---|---|---|---|---|
| AngularSweep | ✗ | ✗ | ✓ | ✗ | ✓ |
| Poly2Vec | ✗ | ✗ | ✗ | ✗ | ✓ |
| Space2Vec | ✗ | ✗ | ✗ | ✗ | ✓ |
| DeepSDF | ✓ | ✓ | ✗ | ✓ | ✗ |
| 2DPE | ✗ | ✗ | ✗ | ✓ | ✓ |
| ShapeEmbed | ✓ | ✓ | ✗ | ✓ | ✗ |
| ShapeDist | ✓ | ✗ | ✓ | ✓ | ✗ |
| *XShapeEnc* (Ours) | ✓ | ✓ | ✓ | ✓ | ✓ |

As shown in the table above, **XShapeEnc** encodes an arbitrary 2D geometric shape associated with a spatial position (e.g., x-, y- coordinate, scale) within a unified framework. It is totally training-free, task-agnostic and frequency-rich, while 
enjoying the advantage of controllable emphasis between shape geometry and shape pose encoding. In summary, it provides flexible encoding:

 | Options | Can XShapeEnc do? |
|---|---|
| Just Shape Rotation Invariant Feature  | **Yes**  |
| Just Shape Rotation Variant Feature | **Yes**  |
| Just Shape Geometry Encoding | **Yes**  |
| Just Shape Pose Encoding | **Yes**  |
| Shape Geometry and Pose Joint Encoding| **Yes** |

### XShapeEnc Pipeline

 <p align="center"><a href="./"><img src=./imgs/XShapeEnc_pipeline_vis.jpg width="70%"></a></p>

 XShapeEnc encoding pipeline is shown in the Figure above. It can independently encode shape geometry, shape pose or geometry-pose jointly. The whole encoding framework is based on Zernike basis. The shape pose requires to be first converted into harmonic pose field so as to be encodable by Zernike basis.

### Experiment

1. Inter-Shape Polygon-Polygon Topological Relation Classification

   <p align="center"><a href="./"><img src=./imgs/topo_rel_vis.jpg width="70%"></a></p>

    We run experiment on spatially grounded polygon pair topological relation classification. As shown in the figure above, we classify 5 main relations: Disjoint, Within, Overlap, Touch and Equal. The polygon shapes are from Singapore and New York building 
    bird-eye-view (BEV) map. The result is shown in the table below,

    | Method | Singapore | New York |
    |---|---:|---:|
    | PointSet | 0.670 | 0.564 |
    | ShapeContexts | 0.581 | 0.525 |
    | AngularSweep | 0.606 | 0.546 |
    | Space2Vec | 0.706 | 0.632 |
    | ResNet18 | 0.674 | 0.753 |
    | ViT | 0.669 | 0.752 |
    | CLIP | 0.700 | 0.779 |
    | Poly2Vec | 0.702 | 0.684 |
    | *XShapeEnc* (Ours) | 0.760 | 0.768 |

    From this table, we can see that **XShapeEnc** maintains highly competitive performance.

2. Invertibility Visualization

    <p align="center"><a href="./"><img src=./imgs/XShapeEnc_invert.jpg width="70%"></a></p>

    We test XShapeEnc's invertibility property by running encoding-to-shape inversion from various commonly used encodng length: 64, 128, 256, 512, 1024, 2048, 4096. The result is shown in the Figure above, from which we can observe that larger encoding length leads to higher-fielity shape recovery.

3. Shape Geometry Clustering Visualization

    <p align="center"><a href="./"><img src=./imgs/geometry_cluster.jpg width="70%"></a></p>

    We test XShapeEnc's shape geometry encoding inter- and intra- geometry discriminability by running t-SNE clustering on augmented 4 complex shapes. As shown in the figure above, XShapeEnc maintains the discriminability while most of the comparing baselines loosing such discriminability.


### Cite XShapeEnc

```bibtex
@inproceedings{yuhheXShapeEnc2026,
title={Training-free Spatially Grounded Geometric Shape Encoding (Technical Report)},
author={He, Yuhang},
booktitle={arXiv:2604.07522},
year={2026}}
```

### Contact

Email: yuhanghe[at]microsoft.com