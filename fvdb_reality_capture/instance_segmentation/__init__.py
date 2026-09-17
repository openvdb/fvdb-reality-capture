# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Scale-conditioned instance segmentation for fVDB Reality Capture."""

from .artifact import (
    ARTIFACT_SCHEMA_VERSION,
    GARfVDBArtifactError,
    GARfVDBArtifactVersionError,
    is_garfvdb_bundle,
)
from .checkpoint import GARFVDB_TRAINING_METHOD
from .config import GARfVDBConfig, GARfVDBTrainingConfig, GARfVDBTransformConfig
from .garfvdb import GARfVDB
from .scene_attribute import GARFVDB_MASK_ATTRIBUTE_NAME, GARfVDBMaskAttribute
from .scene_transforms import GenerateGARfVDBMasks
from .training.segmentation import GARfVDBTrainer

__all__ = [
    "GARfVDB",
    "GARFVDB_TRAINING_METHOD",
    "ARTIFACT_SCHEMA_VERSION",
    "GARfVDBArtifactError",
    "GARfVDBArtifactVersionError",
    "GARfVDBConfig",
    "GARFVDB_MASK_ATTRIBUTE_NAME",
    "GARfVDBMaskAttribute",
    "GenerateGARfVDBMasks",
    "GARfVDBTrainer",
    "GARfVDBTrainingConfig",
    "GARfVDBTransformConfig",
    "is_garfvdb_bundle",
]
