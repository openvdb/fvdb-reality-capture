Enums
=====

The camera enums that fvdb kernels accept are owned by ``fvdb`` and re-exported by
``fvdb_reality_capture`` as the same objects, so values pass between the two packages without
conversion:

- :class:`fvdb.CameraModel` (also available as ``fvdb_reality_capture.CameraModel``)
- :class:`fvdb.RollingShutterType` (also available as ``fvdb_reality_capture.RollingShutterType``)

:class:`ProjectionMethod` and :class:`GaussianRenderMode` select stages of the composable rendering
pipeline in :mod:`fvdb_reality_capture.functional` and are defined here.

.. autoclass:: fvdb_reality_capture.ProjectionMethod
   :members:

.. autoclass:: fvdb_reality_capture.GaussianRenderMode
   :members:
