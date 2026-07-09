# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""GARfVDB training-checkpoint method identity."""

GARFVDB_TRAINING_METHOD = "instance_segmentation.garfvdb"

# Single source of truth for the GARfVDB training-checkpoint format version. The trainer's
# ``state_dict``/``from_state_dict`` and the disk writer's envelope ``method_version`` both derive from
# this constant so the two can never drift.
GARFVDB_TRAINING_METHOD_VERSION = "0.1.0"

__all__ = [
    "GARFVDB_TRAINING_METHOD",
    "GARFVDB_TRAINING_METHOD_VERSION",
]
