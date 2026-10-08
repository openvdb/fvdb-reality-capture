Functional Gaussian Splatting
=============================

.. module:: fvdb_reality_capture.functional

:mod:`fvdb_reality_capture.functional` exposes Gaussian splat rendering as four composable stages
that pass small frozen dataclasses between them. :class:`~fvdb_reality_capture.GaussianSplat3d`
is a thin wrapper that composes these stages; use them directly when you need to insert your own
logic between projection and rasterization, reuse a projection for several renders, or build a
training loop over plain tensors.

The kernels themselves live in :mod:`fvdb.functional` as flat, non-differentiable forward and
backward functions. The stages here attach autograd to them, so gradients flow through every stage
except tile intersection.

.. code-block:: python

   import torch
   import fvdb_reality_capture.functional as F
   from fvdb_reality_capture import CameraModel, GaussianRenderMode

   # Plain tensors: means [N, 3], quats [N, 4], log_scales [N, 3], logit_opacities [N],
   # sh0 [N, 1, 3], shN [N, K - 1, 3], world_to_cam [C, 4, 4], K [C, 3, 3]

   # Stage 1: project the 3D Gaussians into every camera
   projected = F.project_gaussians(
       means, quats, log_scales, world_to_cam, K, image_width=640, image_height=480
   )

   # Stage 2: view-dependent features from spherical harmonics
   features = F.evaluate_gaussian_sh(
       means, sh0, shN, world_to_cam, projected, render_mode=GaussianRenderMode.FEATURES
   )

   # Per-camera opacities, computed once and shared by stages 3 and 4
   opacities = F.compute_gaussian_opacities(logit_opacities, projected)

   # Stage 3: bin Gaussians into image tiles (opacities enable tighter culling)
   tiles = F.intersect_gaussian_tiles(projected, opacities)

   # Stage 4: alpha-blend into images
   images, alphas = F.rasterize_screen_space_gaussians(projected, features, opacities, tiles)

   loss = torch.nn.functional.l1_loss(images, target_images)
   loss.backward()  # gradients reach means, quats, log_scales, logit_opacities, sh0 and shN

The sparse path renders an arbitrary set of pixels per camera. Swap stages 3 and 4 for
:func:`intersect_gaussian_tiles_sparse` and :func:`rasterize_screen_space_gaussians_sparse`; results
come back as :class:`~fvdb.JaggedTensor` in the order of the requested pixels, duplicates included.
The world-space path, :func:`rasterize_world_space_gaussians`, evaluates the 3D Gaussians along
per-pixel rays and is the training path for the unscented projection, whose kernel has no backward
pass.


Types
-----

.. autoclass:: ProjectedGaussians
   :members:

.. autoclass:: GaussianTileIntersection
   :members:

.. autoclass:: SparseGaussianTileIntersection
   :members:


Stage 1: Projection
-------------------

.. autofunction:: project_gaussians

.. autofunction:: resolve_projection_method

.. autofunction:: requires_distortion_coeffs
.. autofunction:: check_distortion_coeffs


Stage 2: Features
-----------------

.. autofunction:: evaluate_gaussian_sh

.. autofunction:: sh_degree_from_coefficients


Stage 3: Tile Intersection
--------------------------

.. autofunction:: intersect_gaussian_tiles

.. autofunction:: intersect_gaussian_tiles_sparse

.. autofunction:: deduplicate_pixels

.. autofunction:: check_tiles_match

.. py:function:: as_pixel_jagged(value)

   ``fvdb.functional.as_pixel_jagged``, re-exported so the sparse pipeline here applies the same
   pixel-selection checks as fvdb's own sparse kernels. Normalizes a ``[C, P, 2]`` tensor or a
   :class:`~fvdb.JaggedTensor` of ``(row, col)`` integer pixels to a JaggedTensor with one list per
   camera, raising ``TypeError`` for non-integer coordinates and ``ValueError`` for malformed shapes.


Stage 4: Rasterization
----------------------

.. autofunction:: rasterize_screen_space_gaussians

.. autofunction:: rasterize_world_space_gaussians

.. autofunction:: rasterize_screen_space_gaussians_sparse

.. autofunction:: compute_gaussian_opacities

.. py:data:: Crop

   ``tuple[int, int, int, int]``: a crop window as ``(origin_w, origin_h, width, height)`` in pixels.

.. autofunction:: validate_crop

.. autofunction:: apply_crop
.. autofunction:: apply_pixel_mask

.. autofunction:: pad_crop

.. autofunction:: pixel_mask_to_tile_mask


Analysis
--------

These do not build an autograd graph.

.. autofunction:: rasterize_num_contributing_gaussians

.. autofunction:: rasterize_contributing_gaussian_ids

.. autofunction:: rasterize_num_contributing_gaussians_sparse

.. autofunction:: rasterize_contributing_gaussian_ids_sparse
