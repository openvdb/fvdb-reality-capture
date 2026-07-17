Gaussian Splatting
==================

The high-level Gaussian splatting API is provided by
``fvdb_reality_capture``. The underlying rendering kernels and supporting
tensor types remain in ``fvdb``.

.. autoclass:: fvdb_reality_capture.ProjectedGaussianSplats
   :members:
   :special-members: __getitem__, __setitem__

.. autoclass:: fvdb_reality_capture.GaussianSplat3d
   :members:
   :special-members: __getitem__, __setitem__

.. autofunction:: fvdb_reality_capture.gaussian_render_jagged

.. autofunction:: fvdb_reality_capture.evaluate_spherical_harmonics

.. autofunction:: fvdb_reality_capture.gaussian_splat_to_view_data
