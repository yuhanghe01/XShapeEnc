## Various experiments center at XShapeEnc

1. Controllable shape geometry and shape pose joint encoding.

   In XShapeEnc, we introduce hyperparameter $\beta$ to specify relative emphasis between shape geometry and shape pose. $\beta\in[0,2]$, where $\beta=1$ means neural emphasis, $\beta\in(1,2]$ means emphasizing geometry, $\beta\in[0,1)$ means emphasizing pose instead. To visualize the effect of emphasis-controllable geometry-pose joint encoding, please run:

   ```python
   python control_geometry_pose.py [--methods XShapeEnc Addition Concate]
                                [--xshapeenc-betas 0.2 1.0 1.8]
                                [--baseline-alphas 0.01 0.04 0.06]
   python control_geometry_pose.py --methods all
   ```

   <p align="center"><a href="./"><img src=../../imgs/control_geopose.jpg width="95%"></a></p>
   
   The above figure shows the different geometry-pose emphasis t-SNE cluterings result, from which we can see different $\beta$ lays different emphasis on geometry and pose, respectively.

2. Shape geometry encoding discriminability

   In XShapeEnc, we take four complex 2D shapes and randomly augment them by randomly rotation, shearing. After running shape geometry encoding by various methods, we cluster these encodings with t-SNE,

   ```python
   python geometry_augment_cluster.py --methods all
   ```

   <p align="center"><a href="./"><img src=../../imgs/geometry_cluster.jpg width="95%"></a></p>
   
   The above figure shows the t-SNE cluterings result of *XShapeEnc* and other baselines, from which we can see *XShapeEnc* maintains both inter-geometry (clear boundary between different shape geometries) and intra-geometry discriminability (belt-line clustering), while most baselines fail to maintain the discriminability.