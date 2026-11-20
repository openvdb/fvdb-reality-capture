# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests for the mesh and point cloud file I/O helpers used by the frgs CLI (frgs points, frgs mesh-basic,
frgs mesh-dlnr, and frgs convert --mesh-path).

Files are read back with trimesh directly (not through the helpers under test) so that a symmetric bug in
reading and writing cannot cancel itself out.
"""

import pathlib
import tempfile
import unittest

import numpy as np
import torch
import trimesh

from fvdb_reality_capture.cli.frgs._common import (
    colors_to_uint8,
    load_mesh,
    save_colored_mesh,
    save_colored_points,
)


def _read_ply_header(path: pathlib.Path) -> str:
    with open(path, "rb") as f:
        data = f.read(4096)
    return data[: data.index(b"end_header")].decode("ascii")


def _test_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    """
    A small mesh designed to expose any processing on save or load: vertices 0 and 4 are exact duplicates
    (merged by trimesh's default ``process=True``), vertex 5 is unreferenced by any face (dropped by mesh
    cleanup), and the faces use mixed winding orders.
    """
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],  # duplicate of vertex 0
            [5.0, 5.0, 5.0],  # unreferenced
            [0.123456789, -98765.4321, 3.0e-7],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [4, 3, 1], [2, 3, 0], [6, 1, 3]], dtype=torch.int64)
    return vertices, faces


class TestColorsToUint8(unittest.TestCase):
    def test_float_colors_are_clamped_scaled_and_truncated(self):
        colors = torch.tensor(
            [
                [0.0, 1.0, 0.5],  # 0.5 * 255 = 127.5 truncates to 127
                [0.999, 1.0e-7, 0.2],  # 0.999 * 255 = 254.7 truncates to 254
                [-0.5, 1.5, 2.0],  # out of range values are clamped
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor([[0, 255, 127], [254, 0, 51], [0, 255, 255]], dtype=torch.uint8)
        result = colors_to_uint8(colors)
        self.assertEqual(result.dtype, torch.uint8)
        self.assertTrue(torch.equal(result, expected))

    def test_float64_colors(self):
        colors = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)
        self.assertTrue(torch.equal(colors_to_uint8(colors), torch.tensor([[0, 127, 255]], dtype=torch.uint8)))

    def test_uint8_colors_pass_through(self):
        colors = torch.arange(256, dtype=torch.uint8).repeat(3, 1).T.contiguous()
        self.assertTrue(torch.equal(colors_to_uint8(colors), colors))

    def test_uint8_float_round_trip_is_lossless(self):
        # Every uint8 value must survive a conversion to float in [0, 1] and back.
        colors = torch.arange(256, dtype=torch.uint8).repeat(3, 1).T.contiguous()
        self.assertTrue(torch.equal(colors_to_uint8(colors.to(torch.float32) / 255.0), colors))


class TestSaveColoredMesh(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp_dir.name) / "mesh.ply"
        self.vertices, self.faces = _test_mesh()
        self.colors = torch.linspace(0.0, 1.0, self.vertices.shape[0] * 3).reshape(-1, 3)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_round_trip_is_exact(self):
        save_colored_mesh(self.path, self.vertices, self.faces, self.colors)

        loaded = trimesh.load(self.path, process=False)
        self.assertIsInstance(loaded, trimesh.Trimesh)
        np.testing.assert_array_equal(loaded.vertices, self.vertices.numpy())
        np.testing.assert_array_equal(loaded.faces, self.faces.numpy())
        np.testing.assert_array_equal(loaded.visual.vertex_colors[:, :3], colors_to_uint8(self.colors).numpy())
        self.assertTrue(np.all(loaded.visual.vertex_colors[:, 3] == 255))

    def test_ply_layout(self):
        save_colored_mesh(self.path, self.vertices, self.faces, self.colors)

        header = _read_ply_header(self.path)
        self.assertIn("format binary_little_endian 1.0", header)
        self.assertIn(f"element vertex {self.vertices.shape[0]}", header)
        self.assertIn(f"element face {self.faces.shape[0]}", header)
        for prop in ("float x", "float y", "float z", "uchar red", "uchar green", "uchar blue"):
            self.assertIn(f"property {prop}", header)

    def test_accepts_uint8_colors_and_float64_vertices(self):
        colors = torch.randint(0, 256, (self.vertices.shape[0], 3), dtype=torch.uint8)
        save_colored_mesh(self.path, self.vertices.to(torch.float64), self.faces, colors)

        loaded = trimesh.load(self.path, process=False)
        np.testing.assert_array_equal(loaded.vertices, self.vertices.numpy())
        np.testing.assert_array_equal(loaded.visual.vertex_colors[:, :3], colors.numpy())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_tensors(self):
        save_colored_mesh(self.path, self.vertices.cuda(), self.faces.cuda(), self.colors.cuda())

        loaded = trimesh.load(self.path, process=False)
        np.testing.assert_array_equal(loaded.vertices, self.vertices.numpy())
        np.testing.assert_array_equal(loaded.faces, self.faces.numpy())


class TestSaveColoredPoints(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp_dir.name) / "points.ply"
        generator = torch.Generator().manual_seed(0)
        self.positions = torch.randn(1000, 3, generator=generator) * 100.0
        self.positions[1] = self.positions[0]  # duplicate points must be kept
        self.colors = torch.randint(0, 256, (1000, 3), dtype=torch.uint8, generator=generator)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_round_trip_is_exact(self):
        save_colored_points(self.path, self.positions, self.colors)

        loaded = trimesh.load(self.path, process=False)
        self.assertIsInstance(loaded, trimesh.PointCloud)
        np.testing.assert_array_equal(loaded.vertices, self.positions.numpy())
        np.testing.assert_array_equal(loaded.colors[:, :3], self.colors.numpy())

    def test_ply_layout(self):
        save_colored_points(self.path, self.positions, self.colors)

        header = _read_ply_header(self.path)
        self.assertIn("format binary_little_endian 1.0", header)
        self.assertIn(f"element vertex {self.positions.shape[0]}", header)
        self.assertNotIn("element face", header)
        for prop in ("float x", "float y", "float z", "uchar red", "uchar green", "uchar blue"):
            self.assertIn(f"property {prop}", header)

    def test_float_colors(self):
        float_colors = self.colors.to(torch.float32) / 255.0
        save_colored_points(self.path, self.positions, float_colors)

        loaded = trimesh.load(self.path, process=False)
        np.testing.assert_array_equal(loaded.colors[:, :3], self.colors.numpy())


class TestLoadMesh(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.vertices, self.faces = _test_mesh()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _check(self, vertices: np.ndarray, faces: np.ndarray):
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(faces.dtype, np.int32)
        np.testing.assert_array_equal(vertices, self.vertices.numpy())
        np.testing.assert_array_equal(faces, self.faces.numpy())

    def test_load_ply(self):
        path = self.root / "mesh.ply"
        trimesh.Trimesh(self.vertices.numpy(), self.faces.numpy(), process=False).export(path)
        self._check(*load_mesh(path))

    def test_load_colored_ply(self):
        path = self.root / "mesh.ply"
        colors = torch.rand(self.vertices.shape[0], 3)
        save_colored_mesh(path, self.vertices, self.faces, colors)
        self._check(*load_mesh(path))

    def _write_obj(self, name: str, face_lines: list[str], header: list[str] | None = None) -> pathlib.Path:
        # Write the OBJ by hand so the test does not depend on trimesh's OBJ exporter.
        path = self.root / name
        lines = list(header or [])
        lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in self.vertices.tolist()]
        lines += face_lines
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_load_obj(self):
        path = self._write_obj("mesh.obj", [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces.tolist()])
        self._check(*load_mesh(path))

    def test_load_obj_with_multiple_materials(self):
        # trimesh's OBJ loader splits meshes by material and duplicates the vertices on the boundaries.
        faces = self.faces.tolist()
        face_lines = []
        for i, (a, b, c) in enumerate(faces):
            face_lines += [f"usemtl material_{i % 2}", f"f {a + 1} {b + 1} {c + 1}"]
        path = self._write_obj("mesh.obj", face_lines, header=["mtllib missing.mtl", "o object"])
        self._check(*load_mesh(path))

    def test_load_obj_face_syntax(self):
        # v/vt, v//vn, v/vt/vn, negative (relative) indices, comments, and line continuations.
        num_vertices = self.vertices.shape[0]
        (a0, b0, c0), (a1, b1, c1), (a2, b2, c2), (a3, b3, c3) = self.faces.tolist()
        face_lines = [
            "vt 0 0",
            "vn 0 0 1",
            f"f {a0 + 1}/1 {b0 + 1}/1 {c0 + 1}/1  # comment",
            f"f {a1 + 1}//1 {b1 + 1}//1 {c1 + 1}//1",
            f"f {a2 + 1}/1/1 \\",
            f"  {b2 + 1}/1/1 {c2 + 1}/1/1",
            f"f {a3 - num_vertices} {b3 - num_vertices} {c3 - num_vertices}",
        ]
        path = self._write_obj("mesh.OBJ", face_lines)
        self._check(*load_mesh(path))

    def test_load_obj_triangulates_polygons(self):
        path = self._write_obj("mesh.obj", ["f 1 2 4 3", "f 2 7 4 5 1"])
        vertices, faces = load_mesh(path)
        np.testing.assert_array_equal(vertices, self.vertices.numpy())
        np.testing.assert_array_equal(faces, [[0, 1, 3], [0, 3, 2], [1, 6, 3], [1, 3, 4], [1, 4, 0]])

    def test_load_obj_rejects_out_of_range_indices(self):
        num_vertices = self.vertices.shape[0]
        for bad_index in (num_vertices + 1, 0, -(num_vertices + 1)):  # OBJ indices are 1-based
            with self.subTest(bad_index=bad_index):
                path = self._write_obj("mesh.obj", [f"f 1 2 {bad_index}"])
                with self.assertRaises(ValueError):
                    load_mesh(path)

    def test_load_obj_without_faces_raises(self):
        path = self._write_obj("points.obj", [])
        with self.assertRaises(ValueError):
            load_mesh(path)

    def test_load_point_cloud_ply_raises(self):
        # trimesh loads a PLY without faces as an empty mesh, which must not be mistaken for a valid mesh.
        path = self.root / "points.ply"
        save_colored_points(path, self.vertices, torch.rand(self.vertices.shape[0], 3))
        with self.assertRaises(ValueError):
            load_mesh(path)


if __name__ == "__main__":
    unittest.main()
