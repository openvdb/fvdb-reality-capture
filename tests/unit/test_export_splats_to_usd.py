# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from fvdb import GaussianSplat3d
from pxr import Usd, UsdVol

from fvdb_reality_capture.tools import export_splats_to_usd


def _make_test_splats(num_gaussians: int = 8, sh_degree: int = 1, seed: int = 0) -> GaussianSplat3d:
    rng = np.random.default_rng(seed)
    num_rest_coeffs = (sh_degree + 1) ** 2 - 1

    means = torch.from_numpy(rng.normal(size=(num_gaussians, 3)).astype(np.float32))
    quats = torch.from_numpy(rng.normal(size=(num_gaussians, 4)).astype(np.float32))
    quats = quats / quats.norm(dim=-1, keepdim=True)
    log_scales = torch.from_numpy(rng.uniform(-2.0, 0.0, size=(num_gaussians, 3)).astype(np.float32))
    logit_opacities = torch.from_numpy(rng.uniform(-2.0, 2.0, size=(num_gaussians,)).astype(np.float32))
    sh0 = torch.from_numpy(rng.uniform(0.0, 1.0, size=(num_gaussians, 1, 3)).astype(np.float32))
    shN = torch.from_numpy(rng.uniform(-0.1, 0.1, size=(num_gaussians, num_rest_coeffs, 3)).astype(np.float32))

    return GaussianSplat3d.from_tensors(means, quats, log_scales, logit_opacities, sh0, shN)


class ExportSplatsToUsdTests(unittest.TestCase):
    def _assert_gaussians_prim_matches_model(self, prim: Usd.Prim, model: GaussianSplat3d) -> None:
        self.assertTrue(prim.IsValid())
        gauss_schema = UsdVol.ParticleField3DGaussianSplat(prim)

        expected_positions = model.means.detach().cpu().numpy()
        positions = np.array(gauss_schema.GetPositionsAttr().Get(), dtype=np.float32)
        np.testing.assert_allclose(positions, expected_positions, atol=1e-5)

        expected_quats = model.quats.detach().cpu().numpy()
        expected_quats = expected_quats / np.linalg.norm(expected_quats, axis=1, keepdims=True)
        orientations = gauss_schema.GetOrientationsAttr().Get()
        quats = np.array([[q.GetReal(), *q.GetImaginary()] for q in orientations], dtype=np.float32)
        np.testing.assert_allclose(quats, expected_quats, atol=1e-5)

        expected_scales = torch.exp(model.log_scales).detach().cpu().numpy()
        scales = np.array(gauss_schema.GetScalesAttr().Get(), dtype=np.float32)
        np.testing.assert_allclose(scales, expected_scales, atol=1e-5)

        expected_opacities = torch.sigmoid(model.logit_opacities).detach().cpu().numpy()
        opacities = np.array(gauss_schema.GetOpacitiesAttr().Get(), dtype=np.float32)
        np.testing.assert_allclose(opacities, expected_opacities, atol=1e-5)

        sh_degree = gauss_schema.GetRadianceSphericalHarmonicsDegreeAttr().Get()
        self.assertEqual(sh_degree, int(model.sh_degree))

        sh_coeffs_attr = gauss_schema.GetRadianceSphericalHarmonicsCoefficientsAttr()
        num_sh_coeffs = sh_coeffs_attr.GetMetadata("elementSize")
        num_gaussians = model.num_gaussians
        sh_coeffs = np.array(sh_coeffs_attr.Get(), dtype=np.float32).reshape(num_gaussians, num_sh_coeffs, 3)

        expected_albedo = model.sh0.detach().cpu().numpy().reshape(num_gaussians, 3)
        np.testing.assert_allclose(sh_coeffs[:, 0, :], expected_albedo, atol=1e-5)

        expected_specular = model.shN.detach().cpu().numpy().reshape(num_gaussians, -1)
        specular = sh_coeffs[:, 1:, :].reshape(num_gaussians, -1)
        np.testing.assert_allclose(specular, expected_specular, atol=1e-5)

    def test_usdc_roundtrip_matches_model(self):
        model = _make_test_splats()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "scene.usdc"
            written_path = export_splats_to_usd(model, out_path)

            self.assertEqual(written_path, out_path)
            self.assertTrue(out_path.is_file())
            with open(out_path, "rb") as f:
                self.assertEqual(f.read(8), b"PXR-USDC")

            stage = Usd.Stage.Open(str(out_path))
            prim = stage.GetPrimAtPath("/World/scene/gaussians")
            self._assert_gaussians_prim_matches_model(prim, model)

    def test_usdz_roundtrip_matches_model(self):
        model = _make_test_splats()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "scene.usdc"
            written_path = export_splats_to_usd(model, out_path, usdz=True)

            self.assertEqual(written_path, out_path.with_suffix(".usdz"))
            self.assertTrue(written_path.is_file())

            stage = Usd.Stage.Open(str(written_path))
            prim = stage.GetPrimAtPath("/World/scene/gaussians")
            self._assert_gaussians_prim_matches_model(prim, model)

    def test_asset_name_override_sets_prim_path(self):
        model = _make_test_splats()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "scene.usdc"
            export_splats_to_usd(model, out_path, asset_name="my_asset")

            stage = Usd.Stage.Open(str(out_path))
            prim = stage.GetPrimAtPath("/World/my_asset/gaussians")
            self._assert_gaussians_prim_matches_model(prim, model)

    def test_usdc_with_mesh_roundtrip_matches_input(self):
        model = _make_test_splats()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "scene.usdc"
            export_splats_to_usd(model, out_path, mesh_vertices=vertices, mesh_faces=faces)

            stage = Usd.Stage.Open(str(out_path))
            self._assert_gaussians_prim_matches_model(stage.GetPrimAtPath("/World/scene/gaussians"), model)

            mesh_prim = stage.GetPrimAtPath("/World/scene/mesh")
            self.assertTrue(mesh_prim.IsValid())
            mesh_points = np.array(mesh_prim.GetAttribute("points").Get(), dtype=np.float32)
            np.testing.assert_allclose(mesh_points, vertices)
            mesh_face_indices = np.array(mesh_prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int32)
            np.testing.assert_array_equal(mesh_face_indices, faces.reshape(-1))

    def test_legacy_requires_usdz(self):
        model = _make_test_splats()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "scene.usdc"
            with self.assertRaises(ValueError):
                export_splats_to_usd(model, out_path, legacy=True)

    def test_legacy_rejects_ecef2enu_rotation(self):
        model = _make_test_splats()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "scene.usdc"
            with self.assertRaises(ValueError):
                export_splats_to_usd(model, out_path, legacy=True, usdz=True, apply_ecef2enu_rotation=True)


if __name__ == "__main__":
    unittest.main()
