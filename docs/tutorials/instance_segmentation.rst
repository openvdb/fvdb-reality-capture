GARfVDB instance segmentation
=============================

GARfVDB learns a scale-conditioned instance feature field from an existing Gaussian reconstruction. The Gaussian
splat is an input and rendering base; the resulting product is a separate, portable instance-segmentation field.

Train from an existing reconstruction
-------------------------------------

First reconstruct the scene, then train GARfVDB against the same posed-image dataset:

.. code-block:: console

   frgs reconstruct --run-name safety-park data/safety_park -o safety-park.ply
   frgs instance-segmentation data/safety_park \
       --reconstruction-path safety-park.ply \
       --out-path safety-park.garfvdb

The reconstruction must have been produced by fVDB Reality Capture and contain its ``normalization_transform``
metadata. Training checkpoints and metrics are written beneath ``frgs_logs`` by default.

Scene transforms and mask supervision
-------------------------------------

GARfVDB uses fVDB Reality Capture's scene-transform pipeline for alignment, point filtering, image
downsampling, image filtering, and cropping. Its SAM2 step is appended as the terminal transform because its mask
rasters and grouping scales describe the final image dimensions and scene coordinate system.

The resulting supervision is attached to the scene as a registered ``GARfVDBMaskAttribute`` named
``garfvdb_masks``. It does not replace ``SfmScene.cache`` or discard other scene attributes. Other segmentation
products can attach their own namespaced attribute and terminal transform, use different SAM2 settings or
post-processing, and coexist on the same scene without adopting GARfVDB's mask/CDF/scale data contract.

Portable artifact
-----------------

``safety-park.garfvdb`` is a versioned directory containing:

* ``manifest.json`` — schema identity and version, configuration, ordered grid metadata, and payload checksums.
* ``encoder.nvdb`` — the ordered multiresolution grid topology and learned per-voxel features.
* ``network.safetensors`` — dense network parameters and scale-quantile lookup tensors.
* ``gaussians.ply`` — the exact filtered Gaussians and their reconstruction metadata.

The bundle contains no PyTorch pickle payload. It can be moved and loaded without the original dataset or
reconstruction. Training and resume still require the source image paths recorded in the checkpoint.

Bundle compatibility
--------------------

Every manifest identifies the format with ``schema: "fvdb_reality_capture.garfvdb"`` and an integer
``schema_version``. The initial format is schema version 1. Loading dispatches to a reader for that exact version;
unknown versions fail before any payload is loaded. A bundle written by a newer release reports that the installed
Reality Capture package must be upgraded. Future format changes can retain an older reader or add an explicit
migration without guessing from package versions or payload contents.

Visualize and query
-------------------

.. code-block:: console

   frgs show safety-park.garfvdb --scale-fraction 0.1 --mask-blend 0.5

The Python API accepts grouping scales in scene units:

.. code-block:: python

   from fvdb_reality_capture.instance_segmentation import GARfVDB

   product = GARfVDB.load("safety-park.garfvdb", device="cuda:0")
   scale = 0.1 * product.max_grouping_scale
   per_gaussian_features = product.gaussian_affinities(scale)

Resume training
---------------

``frgs resume`` reads the generic training-checkpoint envelope and dispatches the stable
``instance_segmentation.garfvdb`` method ID to the GARfVDB resume handler. If the reconstruction moved, pass its new
path explicitly. Portable ``.garfvdb`` products are inference artifacts and cannot be resumed:

.. code-block:: console

   frgs resume frgs_logs/my-run/checkpoints/00010000/train_ckpt.pt \
       --reconstruction-path safety-park.ply \
       --out-path safety-park-resumed.garfvdb
