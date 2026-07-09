.. code-block:: text

   usage: frgs instance-segmentation [-h] [INSTANCE-SEGMENTATION OPTIONS] PATH

   Train a GARfVDB scale-conditioned instance feature field from an existing Reality Capture reconstruction.

   Example:

       frgs instance-segmentation ./colmap_dataset \
           --reconstruction-path scene.ply \
           --out-path scene.garfvdb

   Required inputs are the original posed-image dataset and the matching Reality Capture PLY or checkpoint.
   The output is a portable directory containing NanoVDB encoder grids, safetensors network weights, and the
   exact filtered Gaussian carrier.
