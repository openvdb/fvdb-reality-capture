# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest
import warnings

import cv2
import numpy as np
import torch

from fvdb_reality_capture.sfm_scene import (
    InterpolationMode,
    PerCameraAttribute,
    PerImageRasterAttribute,
    PerImageValueAttribute,
    PerPointAttribute,
    SceneAttribute,
    SfmCache,
    SfmCameraMetadata,
    SfmCameraType,
    SfmPosedImageMetadata,
    SfmScene,
    scene_attribute,
)
from fvdb_reality_capture.sfm_scene.scene_attribute import REGISTERED_SCENE_ATTRIBUTES

# ---------------------------------------------------------------------------
# Helper: build a minimal synthetic SfmScene for testing
# ---------------------------------------------------------------------------


def _make_camera_metadata(width=64, height=48):
    """Build a minimal SfmCameraMetadata."""
    fx, fy = 50.0, 50.0
    cx, cy = width / 2.0, height / 2.0
    return SfmCameraMetadata(
        img_width=width,
        img_height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_type=SfmCameraType.PINHOLE,
        distortion_parameters=np.zeros(0),
    )


def _make_synthetic_scene(num_points=100, num_images=5, num_cameras=2, tmp_dir=None):
    """Build a minimal SfmScene with random data (no real images on disk)."""
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    cache_path = pathlib.Path(tmp_dir)
    cache = SfmCache.get_cache(cache_path, name="test_cache", description="unit test cache")

    cameras = {}
    for cam_id in range(1, num_cameras + 1):
        cameras[cam_id] = _make_camera_metadata()

    images = []
    for i in range(num_images):
        cam_id = (i % num_cameras) + 1
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = np.random.randn(3)
        w2c = np.linalg.inv(c2w)
        images.append(
            SfmPosedImageMetadata(
                world_to_camera_matrix=w2c,
                camera_to_world_matrix=c2w,
                camera_metadata=cameras[cam_id],
                camera_id=cam_id,
                image_path="",
                mask_path="",
                point_indices=None,
                image_id=i,
            )
        )

    points = np.random.randn(num_points, 3).astype(np.float32)
    points_rgb = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)
    points_err = np.random.rand(num_points).astype(np.float32)

    return SfmScene(
        cameras=cameras,
        images=images,
        points=points,
        points_err=points_err,
        points_rgb=points_rgb,
        scene_bbox=None,
        transformation_matrix=None,
        cache=cache,
    )


# ---------------------------------------------------------------------------
# Tier 1: Unit Tests for Attribute Types
# ---------------------------------------------------------------------------


class TestInterpolationMode(unittest.TestCase):
    def test_string_equality(self):
        self.assertEqual(InterpolationMode.AREA, "area")
        self.assertEqual(InterpolationMode.BILINEAR, "bilinear")

    def test_construction_from_string(self):
        self.assertEqual(InterpolationMode("bilinear"), InterpolationMode.BILINEAR)
        self.assertEqual(InterpolationMode("nearest"), InterpolationMode.NEAREST)

    def test_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            InterpolationMode("invalid_mode")


class TestSceneAttributeDecorator(unittest.TestCase):
    def test_decorated_class_in_registry(self):
        self.assertIn("PerPointAttribute", REGISTERED_SCENE_ATTRIBUTES)
        self.assertIn("PerImageValueAttribute", REGISTERED_SCENE_ATTRIBUTES)
        self.assertIn("PerImageRasterAttribute", REGISTERED_SCENE_ATTRIBUTES)
        self.assertIn("PerCameraAttribute", REGISTERED_SCENE_ATTRIBUTES)

    def test_duplicate_registration_replaces(self):
        @scene_attribute
        class _DummyAttr(SceneAttribute):
            @staticmethod
            def type_name():
                return "_DummyAttr"

            def state_dict(self):
                return {}

            @staticmethod
            def from_state_dict(d):
                return _DummyAttr()

        self.assertIn("_DummyAttr", REGISTERED_SCENE_ATTRIBUTES)
        original_cls = REGISTERED_SCENE_ATTRIBUTES["_DummyAttr"]

        @scene_attribute
        class _DummyAttr2(SceneAttribute):
            @staticmethod
            def type_name():
                return "_DummyAttr"

            def state_dict(self):
                return {}

            @staticmethod
            def from_state_dict(d):
                return _DummyAttr2()

        self.assertIs(REGISTERED_SCENE_ATTRIBUTES["_DummyAttr"], _DummyAttr2)
        del REGISTERED_SCENE_ATTRIBUTES["_DummyAttr"]

    def test_undecorated_class_not_in_registry(self):
        class _Unregistered(SceneAttribute):
            @staticmethod
            def type_name():
                return "_Unregistered"

            def state_dict(self):
                return {}

            @staticmethod
            def from_state_dict(d):
                return _Unregistered()

        self.assertNotIn("_Unregistered", REGISTERED_SCENE_ATTRIBUTES)


class TestPerPointAttribute(unittest.TestCase):
    def test_filter_points(self):
        data = np.arange(20).reshape(10, 2).astype(np.float32)
        attr = PerPointAttribute(data)
        mask = np.array([True, False, True, False, True, False, True, False, True, False])
        filtered = attr.on_filter_points(mask)
        self.assertEqual(filtered.data.shape[0], 5)
        np.testing.assert_array_equal(filtered.data, data[mask])

    def test_no_op_hooks(self):
        data = np.random.randn(10, 3).astype(np.float32)
        attr = PerPointAttribute(data)
        self.assertIs(attr.on_filter_images(np.array([True, False])), attr)
        self.assertIs(attr.on_select_images(np.array([0])), attr)
        self.assertIs(attr.on_downsample_images("test", 2, None), attr)
        self.assertIs(attr.on_crop_scene("test", np.zeros(6), None), attr)

    def test_spatial_transform_none(self):
        data = np.random.randn(5, 3).astype(np.float32)
        attr = PerPointAttribute(data, transform_mode="none")
        result = attr.on_spatial_transform(np.eye(4))
        self.assertIs(result, attr)

    def test_spatial_transform_rotate(self):
        from scipy.spatial.transform import Rotation as R

        data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        rot_matrix = R.from_euler("z", 90, degrees=True).as_matrix()
        M = np.eye(4)
        M[:3, :3] = rot_matrix
        attr = PerPointAttribute(data, transform_mode="rotate")
        result = attr.on_spatial_transform(M)
        expected = data @ rot_matrix.T
        np.testing.assert_allclose(result.data, expected, atol=1e-10)

    def test_spatial_transform_rotate_with_scale(self):
        """Similarity transform: normals should rotate but NOT scale."""
        data = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        scale = 5.0
        M = np.eye(4)
        M[:3, :3] = scale * np.eye(3)
        attr = PerPointAttribute(data, transform_mode="rotate")
        result = attr.on_spatial_transform(M)
        np.testing.assert_allclose(result.data, data, atol=1e-10)

    def test_spatial_transform_rigid(self):
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        M = np.eye(4)
        M[:3, 3] = [10.0, 20.0, 30.0]
        attr = PerPointAttribute(data, transform_mode="affine")
        result = attr.on_spatial_transform(M)
        expected = data @ M[:3, :3].T + M[:3, 3]
        np.testing.assert_allclose(result.data, expected, atol=1e-10)

    def test_state_dict_round_trip(self):
        data = np.random.randn(10, 3).astype(np.float32)
        attr = PerPointAttribute(data, transform_mode="rotate")
        sd = attr.state_dict()
        restored = PerPointAttribute.from_state_dict(sd)
        np.testing.assert_array_almost_equal(restored.data, data, decimal=5)
        self.assertEqual(restored.transform_mode, "rotate")

    def test_validate_correct(self):
        data = np.random.randn(10, 3).astype(np.float32)
        attr = PerPointAttribute(data)
        attr.validate("normals", 10, 5, {0, 1})

    def test_validate_wrong_shape(self):
        data = np.random.randn(10, 3).astype(np.float32)
        attr = PerPointAttribute(data)
        with self.assertRaises(ValueError) as ctx:
            attr.validate("normals", 5, 5, {0, 1})
        self.assertIn("normals", str(ctx.exception))


class TestPerImageValueAttribute(unittest.TestCase):
    def test_filter_images(self):
        attr = PerImageValueAttribute([10, 20, 30, 40, 50])
        mask = np.array([True, False, True, False, True])
        filtered = attr.on_filter_images(mask)
        self.assertEqual(filtered.values, [10, 30, 50])

    def test_select_images(self):
        attr = PerImageValueAttribute([10, 20, 30, 40, 50])
        indices = np.array([4, 2, 0])
        selected = attr.on_select_images(indices)
        self.assertEqual(selected.values, [50, 30, 10])

    def test_no_op_hooks(self):
        attr = PerImageValueAttribute([1, 2, 3])
        self.assertIs(attr.on_filter_points(np.array([True, False])), attr)
        self.assertIs(attr.on_spatial_transform(np.eye(4)), attr)
        self.assertIs(attr.on_downsample_images("test", 2, None), attr)
        self.assertIs(attr.on_crop_scene("test", np.zeros(6), None), attr)

    def test_state_dict_round_trip(self):
        attr = PerImageValueAttribute([1.5, 2.5, 3.5])
        sd = attr.state_dict()
        restored = PerImageValueAttribute.from_state_dict(sd)
        self.assertEqual(restored.values, [1.5, 2.5, 3.5])

    def test_validate_correct(self):
        attr = PerImageValueAttribute([1, 2, 3])
        attr.validate("timestamps", 10, 3, {0, 1})

    def test_validate_wrong_length(self):
        attr = PerImageValueAttribute([1, 2, 3])
        with self.assertRaises(ValueError) as ctx:
            attr.validate("timestamps", 10, 5, {0, 1})
        self.assertIn("timestamps", str(ctx.exception))


class TestPerImageRasterAttribute(unittest.TestCase):
    def test_filter_images(self):
        attr = PerImageRasterAttribute(["a.png", "b.png", "c.png"])
        mask = np.array([True, False, True])
        filtered = attr.on_filter_images(mask)
        self.assertEqual(filtered.paths, ["a.png", "c.png"])

    def test_select_images(self):
        attr = PerImageRasterAttribute(["a.png", "b.png", "c.png"])
        indices = np.array([2, 0])
        selected = attr.on_select_images(indices)
        self.assertEqual(selected.paths, ["c.png", "a.png"])

    def test_no_op_hooks(self):
        attr = PerImageRasterAttribute(["a.png"])
        self.assertIs(attr.on_filter_points(np.array([True])), attr)
        self.assertIs(attr.on_spatial_transform(np.eye(4)), attr)

    def test_downsample_images_png(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            img = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
            img_path = pathlib.Path(tmp_dir) / "test_img.png"
            cv2.imwrite(str(img_path), img)

            attr = PerImageRasterAttribute(
                paths=[str(img_path)],
                resize_interpolation=InterpolationMode.AREA,
            )
            result = attr.on_downsample_images("features", 2, cache)
            self.assertEqual(len(result.paths), 1)
            loaded = cv2.imread(result.paths[0], cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape[:2], (16, 32))

    def test_downsample_images_npy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            arr = np.random.randn(32, 64, 8).astype(np.float32)
            arr_path = pathlib.Path(tmp_dir) / "test_arr.npy"
            np.save(str(arr_path), arr)

            attr = PerImageRasterAttribute(
                paths=[str(arr_path)],
                resize_interpolation=InterpolationMode.BILINEAR,
            )
            result = attr.on_downsample_images("depth", 2, cache)
            self.assertEqual(len(result.paths), 1)
            loaded = np.load(result.paths[0])
            self.assertEqual(loaded.shape[0], 16)
            self.assertEqual(loaded.shape[1], 32)

    def test_downsample_images_pt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            tensor = torch.randn(32, 64, 4)
            pt_path = pathlib.Path(tmp_dir) / "test_tensor.pt"
            torch.save(tensor, str(pt_path))

            attr = PerImageRasterAttribute(
                paths=[str(pt_path)],
                resize_interpolation=InterpolationMode.BILINEAR,
            )
            result = attr.on_downsample_images("features", 2, cache)
            loaded = torch.load(result.paths[0], weights_only=False)
            self.assertEqual(loaded.shape, (16, 32, 4))

    def test_downsample_nearest_integer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            tensor = torch.randint(0, 10, (32, 64), dtype=torch.int32)
            pt_path = pathlib.Path(tmp_dir) / "test_int.pt"
            torch.save(tensor, str(pt_path))

            attr = PerImageRasterAttribute(
                paths=[str(pt_path)],
                resize_interpolation=InterpolationMode.NEAREST,
            )
            result = attr.on_downsample_images("labels", 2, cache)
            loaded = torch.load(result.paths[0], weights_only=False)
            self.assertEqual(loaded.dtype, torch.int32)
            self.assertEqual(loaded.shape, (16, 32))

    def test_downsample_integer_wrong_interp_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            tensor = torch.randint(0, 10, (32, 64), dtype=torch.int16)
            pt_path = pathlib.Path(tmp_dir) / "test_int.pt"
            torch.save(tensor, str(pt_path))

            attr = PerImageRasterAttribute(
                paths=[str(pt_path)],
                resize_interpolation=InterpolationMode.AREA,
            )
            with self.assertRaises(TypeError) as ctx:
                attr.on_downsample_images("seg_mask_ids", 2, cache)
            self.assertIn("seg_mask_ids", str(ctx.exception))

    def test_downsample_dict_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            data = {"a": torch.randn(10), "b": torch.randn(10)}
            pt_path = pathlib.Path(tmp_dir) / "test_dict.pt"
            torch.save(data, str(pt_path))

            attr = PerImageRasterAttribute(
                paths=[str(pt_path)],
                resize_interpolation=InterpolationMode.BILINEAR,
            )
            with self.assertRaises(TypeError) as ctx:
                attr.on_downsample_images("seg_data", 2, cache)
            self.assertIn("seg_data", str(ctx.exception))

    def test_downsample_1d_tensor_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            tensor = torch.randn(128)
            pt_path = pathlib.Path(tmp_dir) / "test_1d.pt"
            torch.save(tensor, str(pt_path))

            attr = PerImageRasterAttribute(
                paths=[str(pt_path)],
                resize_interpolation=InterpolationMode.BILINEAR,
            )
            with self.assertRaises(ValueError) as ctx:
                attr.on_downsample_images("seg_scales", 2, cache)
            self.assertIn("seg_scales", str(ctx.exception))

    def test_downsample_pt_numpy_array(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            arr = np.random.randn(32, 64, 2).astype(np.float32)
            pt_path = pathlib.Path(tmp_dir) / "test_nparr.pt"
            torch.save(arr, str(pt_path))

            attr = PerImageRasterAttribute(
                paths=[str(pt_path)],
                resize_interpolation=InterpolationMode.BILINEAR,
            )
            result = attr.on_downsample_images("flow", 2, cache)
            loaded = torch.load(result.paths[0], weights_only=False)
            self.assertEqual(loaded.shape, (16, 32, 2))

    def test_downsample_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            bad_path = pathlib.Path(tmp_dir) / "test_data.xyz"
            bad_path.write_text("dummy")

            attr = PerImageRasterAttribute(
                paths=[str(bad_path)],
                resize_interpolation=InterpolationMode.BILINEAR,
            )
            with self.assertRaises(ValueError) as ctx:
                attr.on_downsample_images("bad_attr", 2, cache)
            self.assertIn(".xyz", str(ctx.exception))

    def test_resize_tensor_higher_rank(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            tensor = torch.randn(32, 64, 3, 2)
            pt_path = pathlib.Path(tmp_dir) / "test_4d.pt"
            torch.save(tensor, str(pt_path))

            attr = PerImageRasterAttribute(
                paths=[str(pt_path)],
                resize_interpolation=InterpolationMode.BILINEAR,
            )
            result = attr.on_downsample_images("high_rank", 2, cache)
            loaded = torch.load(result.paths[0], weights_only=False)
            self.assertEqual(loaded.shape, (16, 32, 3, 2))

    def test_downsample_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test", description="test")
            img = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
            img_path = pathlib.Path(tmp_dir) / "test_img.png"
            cv2.imwrite(str(img_path), img)

            attr = PerImageRasterAttribute(
                paths=[str(img_path)],
                resize_interpolation=InterpolationMode.AREA,
            )
            result1 = attr.on_downsample_images("features", 2, cache)
            result2 = attr.on_downsample_images("features", 2, cache)
            self.assertEqual(result1.paths, result2.paths)

    def test_state_dict_round_trip(self):
        attr = PerImageRasterAttribute(
            paths=["a.npy", "b.npy"],
            resize_interpolation=InterpolationMode.BICUBIC,
        )
        sd = attr.state_dict()
        restored = PerImageRasterAttribute.from_state_dict(sd)
        self.assertEqual(restored.paths, ["a.npy", "b.npy"])
        self.assertEqual(restored.resize_interpolation, InterpolationMode.BICUBIC)

    def test_validate_correct(self):
        attr = PerImageRasterAttribute(["a.png", "b.png"])
        attr.validate("features", 10, 2, {0, 1})

    def test_validate_wrong_length(self):
        attr = PerImageRasterAttribute(["a.png", "b.png"])
        with self.assertRaises(ValueError) as ctx:
            attr.validate("features", 10, 5, {0, 1})
        self.assertIn("features", str(ctx.exception))


class TestPerCameraAttribute(unittest.TestCase):
    def test_all_hooks_no_op(self):
        attr = PerCameraAttribute({0: "IMX586", 1: "OV13B10"})
        self.assertIs(attr.on_filter_points(np.array([True, False])), attr)
        self.assertIs(attr.on_filter_images(np.array([True, False])), attr)
        self.assertIs(attr.on_select_images(np.array([0])), attr)
        self.assertIs(attr.on_spatial_transform(np.eye(4)), attr)
        self.assertIs(attr.on_downsample_images("test", 2, None), attr)
        self.assertIs(attr.on_crop_scene("test", np.zeros(6), None), attr)

    def test_validate_correct(self):
        attr = PerCameraAttribute({1: "x", 2: "y"})
        attr.validate("sensor_info", 10, 5, {1, 2, 3})

    def test_validate_invalid_keys(self):
        attr = PerCameraAttribute({3: "x", 5: "y"})
        with self.assertRaises(ValueError) as ctx:
            attr.validate("sensor_info", 10, 5, {0, 1, 2})
        self.assertIn("sensor_info", str(ctx.exception))

    def test_state_dict_round_trip(self):
        attr = PerCameraAttribute({0: "IMX586", 1: "OV13B10"})
        sd = attr.state_dict()
        restored = PerCameraAttribute.from_state_dict(sd)
        self.assertEqual(restored.values, {0: "IMX586", 1: "OV13B10"})


# ---------------------------------------------------------------------------
# Tier 2: Integration Tests on SfmScene
# ---------------------------------------------------------------------------


class TestSfmSceneAttributes(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name
        self.scene = _make_synthetic_scene(num_points=50, num_images=10, tmp_dir=self.tmp_dir)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_constructor_with_attributes(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        scene = self.scene.replace(attributes={"normals": PerPointAttribute(normals)})
        self.assertTrue(scene.has_attribute("normals"))
        self.assertEqual(len(scene.attributes), 1)

    def test_constructor_without_attributes(self):
        self.assertEqual(len(self.scene.attributes), 0)

    def test_validation_on_construction_wrong_points(self):
        bad_data = np.random.randn(99, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            self.scene.replace(attributes={"normals": PerPointAttribute(bad_data)})

    def test_validation_on_construction_wrong_images(self):
        bad_values = PerImageValueAttribute([1, 2, 3])
        with self.assertRaises(ValueError):
            self.scene.replace(attributes={"timestamps": bad_values})

    def test_validation_empty_attr_name_raises(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        with self.assertRaises(ValueError) as ctx:
            self.scene.replace(attributes={"": PerPointAttribute(normals)})
        self.assertIn("non-empty", str(ctx.exception))

    def test_validation_non_string_attr_name_raises(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        with self.assertRaises(ValueError) as ctx:
            self.scene.replace(attributes={42: PerPointAttribute(normals)})
        self.assertIn("non-empty", str(ctx.exception))

    def test_with_attributes_add(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        scene2 = self.scene.with_attributes(normals=PerPointAttribute(normals))
        self.assertTrue(scene2.has_attribute("normals"))
        self.assertFalse(self.scene.has_attribute("normals"))

    def test_with_attributes_replace(self):
        normals1 = np.random.randn(50, 3).astype(np.float32)
        normals2 = np.random.randn(50, 3).astype(np.float32)
        scene2 = self.scene.with_attributes(normals=PerPointAttribute(normals1))
        scene3 = scene2.with_attributes(normals=PerPointAttribute(normals2))
        np.testing.assert_array_equal(scene3.get_attribute("normals").data, normals2)

    def test_with_attributes_preserves_other(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        values = list(range(10))
        scene2 = self.scene.with_attributes(
            normals=PerPointAttribute(normals),
            timestamps=PerImageValueAttribute(values),
        )
        scene3 = scene2.with_attributes(normals=PerPointAttribute(normals))
        self.assertTrue(scene3.has_attribute("timestamps"))

    def test_without_attributes(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        values = list(range(10))
        scene2 = self.scene.with_attributes(
            normals=PerPointAttribute(normals),
            timestamps=PerImageValueAttribute(values),
        )
        scene3 = scene2.without_attributes("normals")
        self.assertFalse(scene3.has_attribute("normals"))
        self.assertTrue(scene3.has_attribute("timestamps"))

    def test_without_attributes_nonexistent_is_noop(self):
        scene2 = self.scene.without_attributes("nonexistent")
        self.assertEqual(len(scene2.attributes), 0)

    def test_get_attribute(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        scene2 = self.scene.with_attributes(normals=PerPointAttribute(normals))
        attr = scene2.get_attribute("normals")
        np.testing.assert_array_equal(attr.data, normals)

    def test_get_attribute_missing_raises(self):
        with self.assertRaises(KeyError):
            self.scene.get_attribute("nonexistent")

    def test_has_attribute(self):
        self.assertFalse(self.scene.has_attribute("normals"))
        normals = np.random.randn(50, 3).astype(np.float32)
        scene2 = self.scene.with_attributes(normals=PerPointAttribute(normals))
        self.assertTrue(scene2.has_attribute("normals"))

    def test_replace_preserves_attributes(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        scene2 = self.scene.with_attributes(normals=PerPointAttribute(normals))
        scene3 = scene2.replace(points=scene2.points)
        self.assertTrue(scene3.has_attribute("normals"))

    def test_filter_points_propagates(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        values = list(range(10))
        scene2 = self.scene.with_attributes(
            normals=PerPointAttribute(normals),
            timestamps=PerImageValueAttribute(values),
        )
        mask = np.zeros(50, dtype=bool)
        mask[:25] = True
        filtered = scene2.filter_points(mask)
        self.assertEqual(filtered.get_attribute("normals").data.shape[0], 25)
        np.testing.assert_array_equal(filtered.get_attribute("normals").data, normals[:25])
        self.assertEqual(filtered.get_attribute("timestamps").values, values)

    def test_filter_images_propagates(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        values = list(range(10))
        scene2 = self.scene.with_attributes(
            normals=PerPointAttribute(normals),
            timestamps=PerImageValueAttribute(values),
        )
        mask = np.array([True, False, True, False, True, False, True, False, True, False])
        filtered = scene2.filter_images(mask)
        np.testing.assert_array_equal(filtered.get_attribute("normals").data, normals)
        self.assertEqual(filtered.get_attribute("timestamps").values, [0, 2, 4, 6, 8])

    def test_select_images_propagates(self):
        values = list(range(10))
        scene2 = self.scene.with_attributes(
            timestamps=PerImageValueAttribute(values),
        )
        indices = np.array([9, 7, 5])
        selected = scene2.select_images(indices)
        self.assertEqual(selected.get_attribute("timestamps").values, [9, 7, 5])

    def test_apply_transformation_matrix_propagates(self):
        from scipy.spatial.transform import Rotation as R

        normals = np.array([[1.0, 0.0, 0.0]] * 50, dtype=np.float64)
        scene2 = self.scene.replace(
            points=self.scene.points.astype(np.float64),
            attributes={"normals": PerPointAttribute(normals, transform_mode="rotate")},
        )
        rot = R.from_euler("z", 90, degrees=True).as_matrix()
        M = np.eye(4)
        M[:3, :3] = rot
        transformed = scene2.apply_transformation_matrix(M)
        expected = normals @ rot.T
        np.testing.assert_allclose(transformed.get_attribute("normals").data, expected, atol=1e-10)

    def test_apply_transformation_bbox_identity(self):
        bbox = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        scene = self.scene.replace(scene_bbox=bbox)
        result = scene.apply_transformation_matrix(np.eye(4))
        np.testing.assert_allclose(result.scene_bbox, bbox, atol=1e-12)

    def test_apply_transformation_bbox_translation(self):
        bbox = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        scene = self.scene.replace(scene_bbox=bbox)
        M = np.eye(4)
        M[:3, 3] = [10.0, 20.0, 30.0]
        result = scene.apply_transformation_matrix(M)
        expected = np.array([10.0, 20.0, 30.0, 11.0, 21.0, 31.0])
        np.testing.assert_allclose(result.scene_bbox, expected, atol=1e-12)

    def test_apply_transformation_bbox_uniform_scale(self):
        bbox = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        scene = self.scene.replace(scene_bbox=bbox)
        M = np.eye(4)
        M[:3, :3] *= 3.0
        result = scene.apply_transformation_matrix(M)
        expected = np.array([-3.0, -3.0, -3.0, 3.0, 3.0, 3.0])
        np.testing.assert_allclose(result.scene_bbox, expected, atol=1e-12)

    def test_apply_transformation_bbox_rotation_90(self):
        """A 90-degree rotation about Z swaps X and Y extents of the AABB."""
        from scipy.spatial.transform import Rotation as R

        bbox = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 1.0])
        scene = self.scene.replace(scene_bbox=bbox)
        rot = R.from_euler("z", 90, degrees=True).as_matrix()
        M = np.eye(4)
        M[:3, :3] = rot
        result = scene.apply_transformation_matrix(M)
        expected = np.array([-1.0, 0.0, 0.0, 0.0, 2.0, 1.0])
        np.testing.assert_allclose(result.scene_bbox, expected, atol=1e-10)

    def test_apply_transformation_bbox_rotation_45(self):
        """45-degree rotation expands the AABB; the naive 2-corner approach would shrink it."""
        from scipy.spatial.transform import Rotation as R

        bbox = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        scene = self.scene.replace(scene_bbox=bbox)
        rot = R.from_euler("z", 45, degrees=True).as_matrix()
        M = np.eye(4)
        M[:3, :3] = rot
        result = scene.apply_transformation_matrix(M)
        s = np.sqrt(2.0)
        expected_min = np.array([-s / 2, 0.0, 0.0])
        expected_max = np.array([s / 2, s, 0.0])
        np.testing.assert_allclose(result.scene_bbox[:3], expected_min, atol=1e-10)
        np.testing.assert_allclose(result.scene_bbox[3:], expected_max, atol=1e-10)

    def test_apply_transformation_bbox_negative_scale(self):
        """Negative scale flips the bbox; min/max must be recomputed."""
        bbox = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        scene = self.scene.replace(scene_bbox=bbox)
        M = np.eye(4)
        M[0, 0] = -1.0  # flip X
        result = scene.apply_transformation_matrix(M)
        np.testing.assert_allclose(result.scene_bbox[:3], np.array([-4.0, 2.0, 3.0]), atol=1e-12)
        np.testing.assert_allclose(result.scene_bbox[3:], np.array([-1.0, 5.0, 6.0]), atol=1e-12)

    def test_apply_transformation_bbox_none_stays_unbounded(self):
        """When scene_bbox is None (unbounded), transformation preserves that."""
        scene = self.scene.replace(scene_bbox=None)
        M = np.eye(4)
        M[:3, :3] *= 2.0
        result = scene.apply_transformation_matrix(M)
        expected = np.array([-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf])
        np.testing.assert_array_equal(result.scene_bbox, expected)

    def test_serialization_round_trip(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        values = list(range(10))
        scene2 = self.scene.with_attributes(
            normals=PerPointAttribute(normals, transform_mode="rotate"),
            timestamps=PerImageValueAttribute(values),
        )
        sd = scene2.state_dict()
        restored = SfmScene.from_state_dict(sd)
        np.testing.assert_array_almost_equal(restored.get_attribute("normals").data, normals, decimal=5)
        self.assertEqual(restored.get_attribute("normals").transform_mode, "rotate")
        self.assertEqual(restored.get_attribute("timestamps").values, values)

    def test_serialization_unknown_type_raises(self):
        normals = np.random.randn(50, 3).astype(np.float32)
        scene2 = self.scene.with_attributes(normals=PerPointAttribute(normals))
        sd = scene2.state_dict()
        sd["attributes"]["normals"]["type_name"] = "UnknownType"
        with self.assertRaises(ValueError) as ctx:
            SfmScene.from_state_dict(sd)
        self.assertIn("UnknownType", str(ctx.exception))

    def test_serialization_round_trip_raster_attribute(self):
        tmp = pathlib.Path(self.tmp_dir)
        raster_paths = []
        for i in range(10):
            p = tmp / f"raster_{i}.npy"
            np.save(str(p), np.random.randn(48, 64).astype(np.float32))
            raster_paths.append(str(p))

        scene2 = self.scene.with_attributes(
            depth=PerImageRasterAttribute(paths=raster_paths, resize_interpolation=InterpolationMode.NEAREST),
        )
        sd = scene2.state_dict()
        restored = SfmScene.from_state_dict(sd)

        attr = restored.get_attribute("depth")
        self.assertIsInstance(attr, PerImageRasterAttribute)
        self.assertEqual(attr.paths, raster_paths)
        self.assertEqual(attr.resize_interpolation, InterpolationMode.NEAREST)

    def test_serialization_round_trip_camera_attribute(self):
        camera_ids = list(self.scene.cameras.keys())
        cam_values = {cid: {"gamma": 2.2, "response_curve": [1.0, 0.5]} for cid in camera_ids}

        scene2 = self.scene.with_attributes(
            cam_meta=PerCameraAttribute(values=cam_values),
        )
        sd = scene2.state_dict()
        restored = SfmScene.from_state_dict(sd)

        attr = restored.get_attribute("cam_meta")
        self.assertIsInstance(attr, PerCameraAttribute)
        self.assertEqual(attr.values, cam_values)

    def test_serialization_round_trip_all_attribute_types(self):
        tmp = pathlib.Path(self.tmp_dir)
        raster_paths = []
        for i in range(10):
            p = tmp / f"raster_all_{i}.npy"
            np.save(str(p), np.random.randn(48, 64).astype(np.float32))
            raster_paths.append(str(p))

        normals = np.random.randn(50, 3).astype(np.float32)
        values = list(range(10))
        camera_ids = list(self.scene.cameras.keys())
        cam_values = {cid: 2.2 for cid in camera_ids}

        scene2 = self.scene.with_attributes(
            normals=PerPointAttribute(normals, transform_mode="rotate"),
            timestamps=PerImageValueAttribute(values),
            depth=PerImageRasterAttribute(paths=raster_paths),
            cam_gamma=PerCameraAttribute(values=cam_values),
        )
        sd = scene2.state_dict()
        restored = SfmScene.from_state_dict(sd)

        np.testing.assert_array_almost_equal(restored.get_attribute("normals").data, normals, decimal=5)
        self.assertEqual(restored.get_attribute("normals").transform_mode, "rotate")
        self.assertEqual(restored.get_attribute("timestamps").values, values)
        self.assertEqual(restored.get_attribute("depth").paths, raster_paths)
        self.assertEqual(restored.get_attribute("cam_gamma").values, cam_values)


# ---------------------------------------------------------------------------
# Tier 3: Compose ordering validation
# ---------------------------------------------------------------------------


class TestComposeOrderingValidation(unittest.TestCase):
    def test_crop_before_normalize_warns(self):
        from fvdb_reality_capture.transforms import Compose, CropScene, NormalizeScene

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Compose(
                CropScene(bbox=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])),
                NormalizeScene(normalization_type="pca"),
            )
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(
                any("stale" in msg for msg in warning_messages),
                f"Expected warning, got: {warning_messages}",
            )

    def test_normalize_before_crop_no_warning(self):
        from fvdb_reality_capture.transforms import Compose, CropScene, NormalizeScene

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Compose(
                NormalizeScene(normalization_type="pca"),
                CropScene(bbox=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])),
            )
            stale_warnings = [warning for warning in w if "stale" in str(warning.message)]
            self.assertEqual(len(stale_warnings), 0)


# ---------------------------------------------------------------------------
# Tier 4: SfmCache.clear_current_folder
# ---------------------------------------------------------------------------


class TestCacheClearCurrentFolder(unittest.TestCase):
    def test_clear_removes_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test_clear", description="test")
            folder = cache.make_folder("sub", description="sub")
            folder.write_file("a", np.zeros((2, 2), dtype=np.uint8), data_type="png")
            folder.write_file("b", np.zeros((2, 2), dtype=np.uint8), data_type="png")
            self.assertEqual(folder.num_files, 2)
            self.assertTrue(folder.has_file("a"))

            path_a = pathlib.Path(folder.get_file_metadata("a")["path"])
            path_b = pathlib.Path(folder.get_file_metadata("b")["path"])
            self.assertTrue(path_a.exists())
            self.assertTrue(path_b.exists())

            folder.clear_current_folder()
            self.assertEqual(folder.num_files, 0)
            self.assertFalse(folder.has_file("a"))
            self.assertFalse(folder.has_file("b"))
            self.assertFalse(path_a.exists())
            self.assertFalse(path_b.exists())

    def test_folder_survives_clear(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test_survive", description="test")
            folder = cache.make_folder("sub", description="sub")
            folder.write_file("a", np.zeros((2, 2), dtype=np.uint8), data_type="png")
            folder.clear_current_folder()

            folder.write_file("c", np.zeros((4, 4), dtype=np.uint8), data_type="png")
            self.assertEqual(folder.num_files, 1)
            self.assertTrue(folder.has_file("c"))

    def test_downsample_regenerates_after_count_mismatch(self):
        """Cache with wrong file count triggers clear + regeneration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = pathlib.Path(tmp_dir)
            cache = SfmCache.get_cache(tmp, name="test_regen", description="test")

            img1_path = tmp / "img1.png"
            img2_path = tmp / "img2.png"
            cv2.imwrite(str(img1_path), np.zeros((8, 8, 3), dtype=np.uint8))
            cv2.imwrite(str(img2_path), np.full((8, 8, 3), 128, dtype=np.uint8))

            attr_1img = PerImageRasterAttribute(
                paths=[str(img1_path)],
                resize_interpolation=InterpolationMode.AREA,
            )
            result1 = attr_1img.on_downsample_images("regen_test", 2, cache)
            self.assertEqual(len(result1.paths), 1)

            attr_2img = PerImageRasterAttribute(
                paths=[str(img1_path), str(img2_path)],
                resize_interpolation=InterpolationMode.AREA,
            )
            result2 = attr_2img.on_downsample_images("regen_test", 2, cache)
            self.assertEqual(len(result2.paths), 2)
            for p in result2.paths:
                loaded = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.shape[:2], (4, 4))


# ---------------------------------------------------------------------------
# Tier 5: SfmDataset raster attribute + patch_size cropping
# ---------------------------------------------------------------------------


class TestSfmDatasetRasterAttributePatchCrop(unittest.TestCase):
    """Raster attributes must be spatially cropped to match the image when patch_size is set."""

    def _make_scene_with_raster(self, tmp_dir, h=48, w=64):
        tmp = pathlib.Path(tmp_dir)

        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        img_path = tmp / "image_0.png"
        cv2.imwrite(str(img_path), img)

        raster = np.arange(h * w, dtype=np.float32).reshape(h, w)
        raster_path = tmp / "raster_0.npy"
        np.save(str(raster_path), raster)

        camera = _make_camera_metadata(width=w, height=h)
        cameras = {1: camera}
        c2w = np.eye(4, dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        images = [
            SfmPosedImageMetadata(
                world_to_camera_matrix=w2c,
                camera_to_world_matrix=c2w,
                camera_metadata=camera,
                camera_id=1,
                image_path=str(img_path),
                mask_path="",
                point_indices=None,
                image_id=0,
            )
        ]

        cache = SfmCache.get_cache(tmp, name="test", description="test")
        scene = SfmScene(
            cameras=cameras,
            images=images,
            points=np.random.randn(10, 3).astype(np.float32),
            points_err=np.random.rand(10).astype(np.float32),
            points_rgb=np.random.randint(0, 255, (10, 3), dtype=np.uint8),
            scene_bbox=None,
            transformation_matrix=None,
            cache=cache,
            attributes={"raster": PerImageRasterAttribute(paths=[str(raster_path)])},
        )
        return scene, raster

    def test_raster_cropped_to_patch_size(self):
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene, full_raster = self._make_scene_with_raster(tmp_dir)
            patch_size = 16
            dataset = SfmDataset(sfm_scene=scene, patch_size=patch_size, load_attributes=["raster"])

            np.random.seed(42)
            datum = dataset[0]

            self.assertEqual(datum["image"].shape[:2], (patch_size, patch_size))
            self.assertEqual(tuple(datum["raster"].shape[:2]), (patch_size, patch_size))

    def test_raster_content_matches_crop(self):
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            h, w = 48, 64
            scene, full_raster = self._make_scene_with_raster(tmp_dir, h=h, w=w)
            patch_size = 16
            dataset = SfmDataset(sfm_scene=scene, patch_size=patch_size, load_attributes=["raster"])

            np.random.seed(42)
            x = np.random.randint(0, max(w - patch_size, 1))
            y = np.random.randint(0, max(h - patch_size, 1))
            expected_crop = full_raster[y : y + patch_size, x : x + patch_size]

            np.random.seed(42)
            datum = dataset[0]

            np.testing.assert_array_equal(datum["raster"].numpy(), expected_crop)

    def test_raster_full_resolution_without_patch_size(self):
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            h, w = 48, 64
            scene, full_raster = self._make_scene_with_raster(tmp_dir, h=h, w=w)
            dataset = SfmDataset(sfm_scene=scene, load_attributes=["raster"])

            datum = dataset[0]

            self.assertEqual(tuple(datum["raster"].shape), (h, w))
            np.testing.assert_array_equal(datum["raster"].numpy(), full_raster)

    def test_raster_pt_loading(self):
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = pathlib.Path(tmp_dir)
            h, w = 48, 64

            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            img_path = tmp / "image_0.png"
            cv2.imwrite(str(img_path), img)

            raster = torch.randn(h, w, 2)
            raster_path = tmp / "raster_0.pt"
            torch.save(raster, str(raster_path))

            camera = _make_camera_metadata(width=w, height=h)
            c2w = np.eye(4, dtype=np.float64)
            w2c = np.eye(4, dtype=np.float64)
            images = [
                SfmPosedImageMetadata(
                    world_to_camera_matrix=w2c,
                    camera_to_world_matrix=c2w,
                    camera_metadata=camera,
                    camera_id=1,
                    image_path=str(img_path),
                    mask_path="",
                    point_indices=None,
                    image_id=0,
                )
            ]
            cache = SfmCache.get_cache(tmp, name="test", description="test")
            scene = SfmScene(
                cameras={1: camera},
                images=images,
                points=np.random.randn(10, 3).astype(np.float32),
                points_err=np.random.rand(10).astype(np.float32),
                points_rgb=np.random.randint(0, 255, (10, 3), dtype=np.uint8),
                scene_bbox=None,
                transformation_matrix=None,
                cache=cache,
                attributes={"flow": PerImageRasterAttribute(paths=[str(raster_path)])},
            )
            dataset = SfmDataset(sfm_scene=scene, load_attributes=["flow"])
            datum = dataset[0]

            self.assertEqual(tuple(datum["flow"].shape), (h, w, 2))
            torch.testing.assert_close(datum["flow"], raster)

    def _make_scene_with_pt_raster(self, tmp_dir, pt_data, attr_name="raster_pt"):
        """Helper: build a scene where the raster attribute points to a .pt file containing ``pt_data``."""
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        tmp = pathlib.Path(tmp_dir)
        h, w = 48, 64
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        img_path = tmp / "image_0.png"
        cv2.imwrite(str(img_path), img)

        pt_path = tmp / "raster_0.pt"
        torch.save(pt_data, str(pt_path))

        camera = _make_camera_metadata(width=w, height=h)
        c2w = np.eye(4, dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        images = [
            SfmPosedImageMetadata(
                world_to_camera_matrix=w2c,
                camera_to_world_matrix=c2w,
                camera_metadata=camera,
                camera_id=1,
                image_path=str(img_path),
                mask_path="",
                point_indices=None,
                image_id=0,
            )
        ]
        cache = SfmCache.get_cache(tmp, name="test", description="test")
        scene = SfmScene(
            cameras={1: camera},
            images=images,
            points=np.random.randn(10, 3).astype(np.float32),
            points_err=np.random.rand(10).astype(np.float32),
            points_rgb=np.random.randint(0, 255, (10, 3), dtype=np.uint8),
            scene_bbox=None,
            transformation_matrix=None,
            cache=cache,
            attributes={attr_name: PerImageRasterAttribute(paths=[str(pt_path)])},
        )
        return SfmDataset(sfm_scene=scene, load_attributes=[attr_name])

    def test_raster_pt_numpy_loading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            arr = np.random.randn(48, 64, 3).astype(np.float32)
            dataset = self._make_scene_with_pt_raster(tmp_dir, arr)
            datum = dataset[0]
            self.assertIsInstance(datum["raster_pt"], torch.Tensor)
            np.testing.assert_array_equal(datum["raster_pt"].numpy(), arr)

    def test_raster_pt_unsupported_type_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(TypeError) as ctx:
                dataset = self._make_scene_with_pt_raster(tmp_dir, {"bad": "data"})
                dataset[0]
            self.assertIn("raster_pt", str(ctx.exception))
            self.assertIn("dict", str(ctx.exception))

    def test_per_image_value_attribute_loading(self):
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = pathlib.Path(tmp_dir)
            h, w = 48, 64

            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            img_path = tmp / "image_0.png"
            cv2.imwrite(str(img_path), img)

            camera = _make_camera_metadata(width=w, height=h)
            c2w = np.eye(4, dtype=np.float64)
            w2c = np.eye(4, dtype=np.float64)
            images = [
                SfmPosedImageMetadata(
                    world_to_camera_matrix=w2c,
                    camera_to_world_matrix=c2w,
                    camera_metadata=camera,
                    camera_id=1,
                    image_path=str(img_path),
                    mask_path="",
                    point_indices=None,
                    image_id=0,
                )
            ]
            cache = SfmCache.get_cache(tmp, name="test", description="test")
            scene = SfmScene(
                cameras={1: camera},
                images=images,
                points=np.random.randn(10, 3).astype(np.float32),
                points_err=np.random.rand(10).astype(np.float32),
                points_rgb=np.random.randint(0, 255, (10, 3), dtype=np.uint8),
                scene_bbox=None,
                transformation_matrix=None,
                cache=cache,
                attributes={"exposure": PerImageValueAttribute(values=[42.5])},
            )
            dataset = SfmDataset(sfm_scene=scene, load_attributes=["exposure"])
            datum = dataset[0]

            self.assertEqual(datum["exposure"], 42.5)

    def test_reserved_key_collision_raises(self):
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene, _ = self._make_scene_with_raster(tmp_dir)
            for reserved in ("image", "projection", "mask", "sparse_depth"):
                scene_with_attr = scene.replace(
                    attributes={reserved: PerImageValueAttribute(values=[1.0])},
                )
                with self.assertRaises(ValueError) as ctx:
                    SfmDataset(sfm_scene=scene_with_attr, load_attributes=[reserved])
                self.assertIn(reserved, str(ctx.exception))

    def test_non_reserved_key_accepted(self):
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene, _ = self._make_scene_with_raster(tmp_dir)
            dataset = SfmDataset(sfm_scene=scene, load_attributes=["raster"])
            self.assertIn("raster", dataset[0])

    def test_load_attributes_defensive_copy(self):
        from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import SfmDataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene, _ = self._make_scene_with_raster(tmp_dir)
            attrs = ["raster"]
            dataset = SfmDataset(sfm_scene=scene, load_attributes=attrs)
            attrs.append("nonexistent")
            self.assertEqual(len(dataset._load_attributes), 1)


if __name__ == "__main__":
    unittest.main()
