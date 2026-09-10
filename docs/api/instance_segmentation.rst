Instance segmentation
=====================

Portable bundle format
----------------------

``.garfvdb`` manifests use the schema name ``fvdb_reality_capture.garfvdb`` and an integer ``schema_version``.
The current version is available as :data:`ARTIFACT_SCHEMA_VERSION`. Loaders select an exact version-specific
reader and raise :class:`GARfVDBArtifactVersionError` for unsupported versions.

GARfVDB
-------

.. currentmodule:: fvdb_reality_capture.instance_segmentation

.. autoclass:: GARfVDB
   :members:

.. autoclass:: GARfVDBTrainer
   :members: new, from_state_dict, from_checkpoint_file, train, to_product, state_dict

.. autoclass:: GARfVDBConfig
   :members:

.. autoclass:: GARfVDBTrainingConfig
   :members:

.. autoclass:: GARfVDBTransformConfig
   :members:

.. autoclass:: GARfVDBMaskAttribute
   :members:

.. autoclass:: GenerateGARfVDBMasks
   :members:

.. autoexception:: GARfVDBArtifactError

.. autoexception:: GARfVDBArtifactVersionError
