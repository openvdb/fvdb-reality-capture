Training checkpoints
====================

Reality Capture training checkpoints use a method-neutral, versioned container. The container identifies the stable
method ID and keeps method-owned optimizer, dataset, and model state inside ``state``. Product extensions such as
``.garfvdb`` are not used for resume dispatch.

Each method owns its method ID and any adapter for checkpoints written before the container was introduced. The
``frgs resume`` command separately owns resume-handler registration and output defaults, keeping CLI policy out of
the serialization layer.

Container schema versus method
------------------------------

A checkpoint carries two independent versions — one for the outer container and one for the method payload inside
it — on orthogonal axes:

- ``schema`` / ``schema_version`` version the **container format itself**: the fixed set of top-level keys
  (``schema``, ``schema_version``, ``method``, ``method_version``, ``state``) and how they are parsed. ``schema`` is
  a constant string shared by every method, and ``schema_version`` is owned centrally by
  :mod:`fvdb_reality_capture.checkpoints` and selects the reader used to parse the container. Bump it only when the
  container structure changes, which affects every method at once. Nothing method-specific lives at this level.
- ``method`` / ``method_version`` version the **method-owned payload** carried in ``state``. ``method`` is the stable
  identifier of the training method that produced the checkpoint (for example ``radiance_fields.gaussian_splat`` or
  ``instance_segmentation.garfvdb``) and is the dispatch key used on resume. ``method_version`` versions that
  method's ``state`` layout and is owned by the method's package as a single source of truth, so the writer's
  recorded version and the method's ``state_dict``/``from_state_dict`` cannot drift. Bump it only when that one
  method's ``state`` changes, independently of other methods.

Everything method-specific — model weights, optimizer and scheduler state, configuration, global step, and method
metadata — lives inside ``state`` and is opaque to the container layer; only the method interprets it. Because the
axes are independent, the container can stay at one ``schema_version`` while different methods evolve their
``method_version`` separately, and restructuring the container advances ``schema_version`` regardless of any method.

.. automodule:: fvdb_reality_capture.checkpoints
   :members: TrainingCheckpoint, create_training_checkpoint, load_training_checkpoint,
             parse_training_checkpoint, register_legacy_checkpoint_adapter
