# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests for ``frgs reconstruct``. Optimization is mocked out so the tests exercise the command's own logic (chunking,
cropping, and merging) on a small synthetic scene without downloading a dataset or training a model.
"""

import logging
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import tyro

from fvdb_reality_capture import CameraModel, GaussianSplat3d
from fvdb_reality_capture.cli.frgs._reconstruct import Reconstruct
from fvdb_reality_capture.radiance_fields import GaussianSplatReconstruction
from fvdb_reality_capture.sfm_scene import (
    SfmCache,
    SfmCameraMetadata,
    SfmPosedImageMetadata,
    SfmScene,
)
from fvdb_reality_capture.transforms import CropScene

_FROM_SFM_SCENE = "fvdb_reality_capture.cli.frgs._reconstruct.GaussianSplatReconstruction.from_sfm_scene"


def _make_scene(cache_dir: pathlib.Path, points: np.ndarray) -> SfmScene:
    """A minimal SfmScene with the given points and a few cameras looking at them (no images on disk)."""
    camera = SfmCameraMetadata(
        img_width=64,
        img_height=48,
        fx=50.0,
        fy=50.0,
        cx=32.0,
        cy=24.0,
        camera_model=CameraModel.PINHOLE,
        distortion_coeffs=np.zeros(0, dtype=np.float32),
    )
    images = []
    for i, x in enumerate((-1.5, 0.0, 1.5)):
        # Cameras sit at z = -5 looking down +z at the points.
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = (x, 0.0, -5.0)
        images.append(
            SfmPosedImageMetadata(
                world_to_camera_matrix=np.linalg.inv(camera_to_world),
                camera_to_world_matrix=camera_to_world,
                camera_metadata=camera,
                camera_id=1,
                image_path="",
                mask_path="",
                point_indices=None,
                image_id=i,
            )
        )
    return SfmScene(
        cameras={1: camera},
        images=images,
        points=points,
        points_err=np.zeros(points.shape[0], dtype=np.float32),
        points_rgb=np.full((points.shape[0], 3), 128, dtype=np.uint8),
        scene_bbox=None,
        transformation_matrix=None,
        cache=SfmCache.get_cache(cache_dir, name="test_cache", description="frgs reconstruct test"),
    )


def _command(root: pathlib.Path, out_path: pathlib.Path, *extra_args: str) -> Reconstruct:
    command = tyro.cli(
        Reconstruct,
        args=[str(root / "dataset"), "--out-path", str(out_path), "--device", "cuda:0", *extra_args],
    )
    command.logger = logging.getLogger(__name__)
    return command


def _splats_at(points: np.ndarray, device: torch.device) -> GaussianSplat3d:
    num_points = points.shape[0]
    return GaussianSplat3d.from_tensors(
        means=torch.from_numpy(points).to(device=device, dtype=torch.float32),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_points, 1),
        log_scales=torch.full((num_points, 3), -3.0, device=device),
        logit_opacities=torch.zeros(num_points, device=device),
        sh0=torch.zeros((num_points, 1, 3), device=device),
        shN=torch.zeros((num_points, 0, 3), device=device),
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class ChunkedReconstructionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        # Two clusters of points, one on each side of x = 0, so a 2x1x1 split puts one cluster in each chunk. The
        # split falls at the midpoint of the points' x extent, so the default 10% overlap stays within |x| < 0.6 for
        # any draw, and all points have |x| >= 1.
        rng = np.random.default_rng(0)
        left = rng.uniform((-2.0, -1.0, -1.0), (-1.0, 1.0, 1.0), size=(10, 3))
        right = rng.uniform((1.0, -1.0, -1.0), (2.0, 1.0, 1.0), size=(15, 3))
        self.points = np.concatenate([left, right]).astype(np.float64)
        self.scene = _make_scene(self.root / "cache", self.points)
        self.device = torch.device("cuda:0")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _command(self, out_path: pathlib.Path, *extra_args: str) -> Reconstruct:
        return _command(self.root, out_path, "--nchunks", "2", "1", "1", *extra_args)

    def _run(
        self, command: Reconstruct, empty_chunks: tuple[int, ...] = (), scene: SfmScene | None = None
    ) -> list[SfmScene]:
        """
        Run the chunked reconstruction with training mocked out: the "optimized" model for a chunk has one splat at
        each of its points, or no splats if the chunk is the n-th one trained and n is in ``empty_chunks``. Returns
        the scenes of the chunks that were trained.
        """
        chunk_scenes: list[SfmScene] = []

        def fake_from_sfm_scene(sfm_scene: SfmScene, **kwargs):
            points = sfm_scene.points[:0] if len(chunk_scenes) in empty_chunks else sfm_scene.points
            chunk_scenes.append(sfm_scene)
            runner = mock.MagicMock()
            runner.model = _splats_at(points, self.device)
            runner.reconstruction_metadata = {}
            return runner

        with mock.patch(_FROM_SFM_SCENE, side_effect=fake_from_sfm_scene):
            command._run_chunked_reconstruction(
                self.scene if scene is None else scene, writer=mock.MagicMock(), viz_scene=None
            )
        return chunk_scenes

    def _merged_means(self, out_path: pathlib.Path) -> np.ndarray:
        merged, _ = GaussianSplat3d.from_ply(out_path, device=self.device)
        return merged.means.cpu().numpy()

    def test_chunks_are_cropped_and_merged(self):
        out_path = self.root / "merged.ply"
        command = self._command(out_path)
        chunk_scenes = self._run(command)

        # Each chunk holds only the points in its bounding box, and every point is in exactly one chunk.
        self.assertEqual(len(chunk_scenes), 2)
        bboxes = command.get_crop_bboxes(self.scene)
        for chunk_scene, bbox in zip(chunk_scenes, bboxes):
            self.assertGreater(chunk_scene.points.shape[0], 0)
            self.assertTrue(np.all(chunk_scene.points >= np.asarray(bbox[:3]) - 1e-6))
            self.assertTrue(np.all(chunk_scene.points <= np.asarray(bbox[3:]) + 1e-6))
        self.assertEqual(sum(s.points.shape[0] for s in chunk_scenes), self.points.shape[0])

        # The merged model holds the splats from every chunk.
        merged_means = self._merged_means(out_path)
        self.assertEqual(merged_means.shape[0], self.points.shape[0])
        np.testing.assert_allclose(
            np.sort(merged_means, axis=0), np.sort(self.points, axis=0).astype(np.float32), rtol=0, atol=1e-6
        )

    def test_empty_chunk_is_skipped(self):
        out_path = self.root / "merged.ply"
        chunk_scenes = self._run(self._command(out_path), empty_chunks=(0,))

        merged_means = self._merged_means(out_path)
        expected = chunk_scenes[1].points.astype(np.float32)
        self.assertEqual(merged_means.shape[0], expected.shape[0])
        np.testing.assert_allclose(np.sort(merged_means, axis=0), np.sort(expected, axis=0), rtol=0, atol=1e-6)

    def test_all_chunks_empty_raises(self):
        out_path = self.root / "merged.ply"
        with self.assertRaisesRegex(RuntimeError, "None of the 2 chunks produced any gaussians"):
            self._run(self._command(out_path), empty_chunks=(0, 1))
        self.assertFalse(out_path.exists())

    def test_sparse_chunk_is_skipped_before_training(self):
        # The right-hand chunk has too few points to initialize a model, so it must not be trained at all.
        points = np.concatenate([self.points[:10], [[1.5, 0.0, 0.0], [2.0, 0.5, 0.5]]])
        scene = _make_scene(self.root / "sparse_cache", points)
        out_path = self.root / "merged.ply"
        chunk_scenes = self._run(self._command(out_path), scene=scene)

        self.assertEqual(len(chunk_scenes), 1)
        merged_means = self._merged_means(out_path)
        np.testing.assert_allclose(
            np.sort(merged_means, axis=0), np.sort(points[:10], axis=0).astype(np.float32), rtol=0, atol=1e-6
        )

    def test_too_few_points_to_initialize_raises(self):
        # The single (unchunked) path and library callers get an error instead of a model with infinite scales.
        scene = _make_scene(self.root / "tiny_cache", self.points[:3])
        with self.assertRaisesRegex(ValueError, "at least 4 are required"):
            GaussianSplatReconstruction.from_sfm_scene(scene, writer=mock.MagicMock(), device=self.device)

    def test_single_reconstruction_with_zero_gaussians_raises(self):
        out_path = self.root / "single.ply"
        command = _command(self.root, out_path)
        runner = mock.MagicMock()
        runner.model = _splats_at(self.points[:0], self.device)
        with mock.patch(_FROM_SFM_SCENE, return_value=runner):
            with self.assertRaisesRegex(RuntimeError, "0 gaussians"):
                command._run_single_reconstruction(self.scene, writer=mock.MagicMock(), viz_scene=None)
        runner.save_ply.assert_not_called()
        self.assertFalse(out_path.exists())


class CropBboxesTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _command(self, *args: str) -> Reconstruct:
        return _command(self.root, self.root / "out.ply", *args)

    def _assert_every_point_in_a_chunk(self, points: np.ndarray, command: Reconstruct):
        scene = _make_scene(self.root / f"cache_{points.dtype}", points)
        bboxes = command.get_crop_bboxes(scene)
        in_any_chunk = np.zeros(points.shape[0], dtype=bool)
        for bbox in bboxes:
            cropped = CropScene(bbox=bbox)(scene).points
            in_any_chunk |= (points[:, None, :] == cropped[None, :, :]).all(axis=-1).any(axis=1)
        self.assertTrue(in_any_chunk.all(), f"points in no chunk: {points[~in_any_chunk]}")

    def test_zero_overlap_keeps_every_point(self):
        # Points on the scene's outer faces and one exactly on the boundary between the two chunks (x = 0).
        points = np.array(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-0.5, 0.2, 0.3], [0.5, -0.2, -0.3]],
            dtype=np.float64,
        )
        command = self._command("--nchunks", "2", "1", "1", "--chunk-overlap-pct", "0")
        self._assert_every_point_in_a_chunk(points, command)

    def test_zero_overlap_keeps_every_point_at_large_float32_coordinates(self):
        # ECEF-scale coordinates in float32, where a margin relative to the scene extent alone would round away.
        offset = np.array([6.0e6, 1.0e6, 3.0e6])
        points = (offset + np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.5, 0.5]])).astype(np.float32)
        command = self._command("--nchunks", "2", "2", "2", "--chunk-overlap-pct", "0")
        self._assert_every_point_in_a_chunk(points, command)

    def test_invalid_chunking_raises(self):
        scene = _make_scene(self.root / "cache", np.zeros((4, 3)))
        for args in (("--nchunks", "0", "1", "1"), ("--chunk-overlap-pct", "1.0"), ("--chunk-overlap-pct", "-0.1")):
            with self.subTest(args=args):
                command = self._command(*args)
                with self.assertRaises(ValueError):
                    command.get_crop_bboxes(scene)
                # execute() rejects the flags before loading the dataset or creating any output.
                with mock.patch("fvdb_reality_capture.cli.frgs._reconstruct.load_sfm_scene") as load_sfm_scene:
                    with self.assertRaises(ValueError):
                        command.execute()
                    load_sfm_scene.assert_not_called()


if __name__ == "__main__":
    unittest.main()
