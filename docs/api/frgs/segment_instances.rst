.. code-block:: text

   usage: frgs segment-instances [-h] [SEGMENT-INSTANCES OPTIONS] PATH

   Train a GARfVDB scale-conditioned instance feature field from an existing Reality Capture reconstruction.

   Example:

       frgs segment-instances ./colmap_dataset \
           --reconstruction-path scene.ply \
           --out-path scene.garfvdb

   Required inputs are the original posed-image dataset and the matching Reality Capture PLY or checkpoint.
   The output is a portable directory containing NanoVDB encoder grids, safetensors network weights, and the
   exact filtered Gaussian model.
