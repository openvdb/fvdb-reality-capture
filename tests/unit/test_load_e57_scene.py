# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest

import cv2
import numpy as np

from fvdb_reality_capture.sfm_scene import SfmCache
from fvdb_reality_capture.sfm_scene._load_e57_scene import _load_e57_scan


class _FakeE57Node:
    """Minimal stand-in for a pye57/libE57 structure, vector or leaf node."""

    def __init__(self, value=None, children=None):
        self._value = value
        self._children = children if children is not None else {}

    def isDefined(self, key):
        return key in self._children

    def __getitem__(self, key):
        return self._children[key]

    def __len__(self):
        return len(self._children)

    def value(self):
        return self._value


class _FakeE57Blob:
    """Minimal stand-in for a libE57 BlobNode holding JPEG bytes."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def byteCount(self):
        return self._data.size

    def read(self, buffer, start, count):
        buffer[:count] = self._data[start : start + count]


class _FakeE57ImageFile:
    def __init__(self, root: _FakeE57Node):
        self._root = root

    def root(self):
        return self._root


class _FakeE57File:
    """Minimal stand-in for pye57.E57 with one scan and one pinhole image looking down at it."""

    def __init__(self, points: np.ndarray):
        self.path = "fake.e57"
        self.scan_count = 1
        self._points = points

        node = _FakeE57Node
        guid = "scan-0"
        self.root = {"data3D": [node(children={"guid": node(guid)})]}

        ok, jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        assert ok
        pinhole = node(
            children={
                "focalLength": node(0.01),
                "pixelWidth": node(1e-5),
                "pixelHeight": node(1e-5),
                "principalPointX": node(4.0),
                "principalPointY": node(4.0),
                "imageWidth": node(8),
                "imageHeight": node(8),
                "jpegImage": _FakeE57Blob(jpeg.ravel()),
            }
        )
        # Camera 10 m above the points with an identity rotation (E57 cameras look down -Z)
        camera_position = points.mean(axis=0) + np.array([0.0, 0.0, 10.0])
        pose = node(
            children={
                "rotation": node(children={"w": node(1.0), "x": node(0.0), "y": node(0.0), "z": node(0.0)}),
                "translation": node(children={k: node(v) for k, v in zip("xyz", camera_position)}),
            }
        )
        image = node(children={"pinholeRepresentation": pinhole, "pose": pose, "associatedData3DGuid": node(guid)})
        self.image_file = _FakeE57ImageFile(node(children={"images2D": node(children=[image])}))

    def read_scan(self, scan_idx, **kwargs):
        num_points = self._points.shape[0]
        return {
            "cartesianX": self._points[:, 0],
            "cartesianY": self._points[:, 1],
            "cartesianZ": self._points[:, 2],
            "colorRed": np.full(num_points, 255, dtype=np.uint8),
            "colorGreen": np.zeros(num_points, dtype=np.uint8),
            "colorBlue": np.zeros(num_points, dtype=np.uint8),
            "intensity": np.ones(num_points, dtype=np.float64),
        }


class LoadE57SceneTest(unittest.TestCase):
    def test_points_keep_float64_precision(self):
        ecef_base = np.array([1100000.0, -4780000.0, 4050000.0], dtype=np.float64)
        expected_points = ecef_base + np.array([[0.01, 0.02, 0.03], [0.04, -0.05, 0.06], [-0.07, 0.08, -0.09]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SfmCache.get_cache(pathlib.Path(tmp_dir), name="test_cache", description="unit test cache")
            _, _, points, _, _ = _load_e57_scan(
                _FakeE57File(expected_points),  # type: ignore
                camera_metadata={},
                image_metadata=[],
                cum_num_points=0,
                cache=cache,
                total_images=1,
            )

        self.assertEqual(points.dtype, np.float64)
        np.testing.assert_array_equal(points, expected_points)


if __name__ == "__main__":
    unittest.main()
