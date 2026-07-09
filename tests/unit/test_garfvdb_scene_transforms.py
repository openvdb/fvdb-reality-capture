# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pathlib
import tempfile
from unittest import mock

import cv2
import numpy as np
import pytest
import torch
from fvdb import CameraModel

from fvdb_reality_capture.instance_segmentation.config import GARfVDBTransformConfig
from fvdb_reality_capture.instance_segmentation.scene_attribute import (
    GARFVDB_MASK_ATTRIBUTE_NAME,
    GARFVDB_MASK_DATA_SCHEMA_VERSION,
    GARfVDBMaskAttribute,
)
from fvdb_reality_capture.instance_segmentation.scene_transforms import GenerateGARfVDBMasks
from fvdb_reality_capture.instance_segmentation.training.dataset import SegmentationDataset
from fvdb_reality_capture.sfm_scene import (
    PerImageValueAttribute,
    SfmCache,
    SfmCameraMetadata,
    SfmPosedImageMetadata,
    SfmScene,
)
from fvdb_reality_capture.transforms import CropScene, Identity, SceneTransformConfig, UndistortImages


def _make_scene(
    directory: pathlib.Path,
    num_images: int = 2,
    *,
    camera_model: CameraModel = CameraModel.PINHOLE,
    distortion_coeffs: np.ndarray | None = None,
) -> SfmScene:
    cache = SfmCache.get_cache(directory, name="garfvdb_transform_test", description="GARfVDB transform test")
    if distortion_coeffs is None:
        distortion_coeffs = np.zeros(0, dtype=np.float32)
    camera = SfmCameraMetadata(
        img_width=8,
        img_height=6,
        fx=6.0,
        fy=6.0,
        cx=4.0,
        cy=3.0,
        camera_model=camera_model,
        distortion_coeffs=distortion_coeffs,
    )
    image_paths = []
    for index in range(num_images):
        image_path = directory / f"image_{index}.png"
        assert cv2.imwrite(str(image_path), np.zeros((6, 8, 3), dtype=np.uint8))
        image_paths.append(image_path)
    images = [
        SfmPosedImageMetadata(
            world_to_camera_matrix=np.eye(4),
            camera_to_world_matrix=np.eye(4),
            camera_metadata=camera,
            camera_id=1,
            image_path=str(image_paths[index]),
            mask_path="",
            point_indices=None,
            image_id=index,
        )
        for index in range(num_images)
    ]
    return SfmScene(
        cameras={1: camera},
        images=images,
        points=np.zeros((0, 3), dtype=np.float32),
        points_err=np.zeros(0, dtype=np.float32),
        points_rgb=np.zeros((0, 3), dtype=np.uint8),
        scene_bbox=None,
        transformation_matrix=None,
        cache=cache,
        attributes={"other_method": PerImageValueAttribute(list(range(num_images)))},
    )


def _mask_data(index: int) -> dict[str, torch.Tensor | int]:
    return {
        "schema_version": GARFVDB_MASK_DATA_SCHEMA_VERSION,
        "scales": torch.tensor([0.1 + index]),
        "pixel_to_mask_id": torch.zeros((6, 8, 1), dtype=torch.int16),
        "mask_cdf": torch.ones((6, 8, 1)),
    }


def test_garfvdb_mask_attribute_round_trip_and_scene_operations():
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        paths = []
        for index in range(3):
            path = root / f"mask_{index}.pt"
            torch.save(_mask_data(index), path)
            paths.append(path)

        attribute = GARfVDBMaskAttribute(paths, provenance={"generator": "test"})
        restored = GARfVDBMaskAttribute.from_state_dict(attribute.state_dict())
        assert restored.paths == [str(path.absolute()) for path in paths]
        assert restored.provenance == {"generator": "test"}
        assert torch.equal(restored.load(1)["scales"], torch.tensor([1.1]))
        assert restored.on_filter_images(np.array([True, False, True])).paths == [
            str(paths[0].absolute()),
            str(paths[2].absolute()),
        ]
        assert restored.on_select_images(np.array([2, 0])).paths == [
            str(paths[2].absolute()),
            str(paths[0].absolute()),
        ]

        with pytest.raises(ValueError, match="after all image transforms"):
            restored.on_downsample_images("garfvdb_masks", 2, None)
        with pytest.raises(ValueError, match="after all crop transforms"):
            restored.on_crop_scene("garfvdb_masks", np.zeros(6), None)
        with pytest.raises(ValueError, match="scene alignment"):
            restored.on_spatial_transform(np.eye(4))


def test_cached_mask_transform_attaches_attribute_without_replacing_scene_state():
    with tempfile.TemporaryDirectory() as directory:
        scene = _make_scene(pathlib.Path(directory))
        carrier_hash = hashlib.sha256(b"test carrier").hexdigest()
        transform = GenerateGARfVDBMasks(
            gs3d=None,
            gs3d_hash=carrier_hash,
            checkpoint="large",
            points_per_side=4,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8,
            device="cpu",
        )
        cache = scene.cache.make_folder(
            f"garfvdb_masks_v1_{carrier_hash}_p4_i80_s80",
            description="cached GARfVDB masks",
        )
        for index in range(scene.num_images):
            cache.write_file(
                name=f"masks_{index:03d}",
                data=_mask_data(index),
                data_type="pt",
                metadata={
                    "points_per_side": 4,
                    "pred_iou_thresh": 0.8,
                    "stability_score_thresh": 0.8,
                    "gs3d_hash": carrier_hash,
                },
            )

        transformed = transform(scene)
        assert transformed.cache.current_folder_id == scene.cache.current_folder_id
        assert transformed.has_attribute("other_method")
        assert transformed.has_attribute(GARFVDB_MASK_ATTRIBUTE_NAME)
        attribute = transformed.get_attribute(GARFVDB_MASK_ATTRIBUTE_NAME)
        assert isinstance(attribute, GARfVDBMaskAttribute)
        assert len(attribute.paths) == scene.num_images

        dataset = SegmentationDataset(transformed, cache_loaded_masks=False, cache_images=False)
        assert torch.equal(dataset.scales, torch.tensor([0.1, 1.1]))

        restored_scene = SfmScene.from_state_dict(transformed.state_dict())
        restored_attribute = restored_scene.get_attribute(GARFVDB_MASK_ATTRIBUTE_NAME)
        assert isinstance(restored_attribute, GARfVDBMaskAttribute)
        assert restored_attribute.paths == attribute.paths


def test_standard_scene_pipeline_places_product_transform_last():
    common = SceneTransformConfig(crop_bbox=(-1, -1, -1, 1, 1, 1))
    terminal = Identity()
    pipeline = common.build_scene_transform(
        alignment_transform=Identity(),
        terminal_transforms=[terminal],
    )
    assert isinstance(pipeline.transforms[-2], CropScene)
    assert pipeline.transforms[-1] is terminal

    with mock.patch(
        "fvdb_reality_capture.instance_segmentation.config.GenerateGARfVDBMasks",
        return_value=terminal,
    ):
        garfvdb_pipeline = GARfVDBTransformConfig(crop_bbox=(-1, -1, -1, 1, 1, 1)).build_scene_transforms(
            gs3d=mock.Mock(),
            normalization_transform=None,
        )
    assert isinstance(garfvdb_pipeline.transforms[-3], CropScene)
    assert isinstance(garfvdb_pipeline.transforms[-2], UndistortImages)
    assert garfvdb_pipeline.transforms[-1] is terminal


def test_garfvdb_mask_generation_uses_materialized_pinhole_images():
    with tempfile.TemporaryDirectory() as directory:
        scene = _make_scene(pathlib.Path(directory), num_images=1)
        carrier = mock.Mock()
        carrier.means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with mock.patch(
            "fvdb_reality_capture.instance_segmentation.scene_transforms.image_segmentation_masks.SAM2Model"
        ):
            transform = GenerateGARfVDBMasks(gs3d=carrier, device="cpu")
        mask_data = _mask_data(0)
        with mock.patch.object(
            transform,
            "_generate_segmentation_mask",
            return_value=(mask_data["scales"], mask_data["pixel_to_mask_id"], mask_data["mask_cdf"]),
        ) as generate:
            transformed = transform(scene)

        assert transformed.has_attribute(GARFVDB_MASK_ATTRIBUTE_NAME)
        generated_image = generate.call_args.args[1]
        assert generated_image.shape == (6, 8, 3)


def test_garfvdb_mask_generation_rejects_distorted_scene():
    with tempfile.TemporaryDirectory() as directory:
        distortion_coeffs = np.zeros(12, dtype=np.float32)
        distortion_coeffs[0] = 0.1
        scene = _make_scene(
            pathlib.Path(directory),
            num_images=1,
            camera_model=CameraModel.OPENCV_RADTAN_5,
            distortion_coeffs=distortion_coeffs,
        )
        transform = GenerateGARfVDBMasks(gs3d=None, gs3d_hash="unused", device="cpu")

        with pytest.raises(ValueError, match="Apply UndistortImages"):
            transform(scene)


def test_segmentation_dataset_requires_namespaced_garfvdb_attribute():
    with tempfile.TemporaryDirectory() as directory:
        scene = _make_scene(pathlib.Path(directory), num_images=1)
        with pytest.raises(ValueError, match=GARFVDB_MASK_ATTRIBUTE_NAME):
            SegmentationDataset(scene)
