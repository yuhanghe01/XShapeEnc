## Various experiments center at XShapeEnc

1. Controllable shape geometry and shape pose joint encoding.

   In XShapeEnc, we introduce hyperparameter $\beta$ to specify relative emphasis between shape geometry and shape pose. $\beta\in[0,2]$, where $\beta=1$ means neural emphasis, $\beta\in(1,2]$ means emphasizing geometry, $\beta\in[0,1)$ means emphasizing pose instead. To visualize the effect of emphasis-controllable geometry-pose joint encoding, please run:

   ```python
   python control_geometry_pose.py
   ```