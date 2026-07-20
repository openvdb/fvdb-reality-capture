Training checkpoints
====================

Reality Capture training checkpoints use a method-neutral, versioned container. The container identifies the stable
method ID and keeps method-owned optimizer, dataset, and model state inside ``state``. Product extensions such as
``.garfvdb`` are not used for resume dispatch.

Each method owns its method ID and any adapter for checkpoints written before the container was introduced. The
``frgs resume`` command separately owns resume-handler registration and output defaults, keeping CLI policy out of
the serialization layer.

.. automodule:: fvdb_reality_capture.checkpoints
   :members: TrainingCheckpoint, create_training_checkpoint, load_training_checkpoint,
             parse_training_checkpoint, register_legacy_checkpoint_adapter
