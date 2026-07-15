# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import gzip
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import msgpack
import numpy as np
import torch
from fvdb import GaussianSplat3d
from pxr import Usd, UsdVol

from fvdb_reality_capture.tools import export_splats_to_usd
from fvdb_reality_capture.tools._export_splats_to_usd import _resize_sh_coefficients, build_legacy_gaussians_payload


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


def _decode_nurec_payload(raw: bytes) -> dict:
    """Decode the gzipped-msgpack legacy NuRec payload and pull out its SH layout metadata."""
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        unpacked = msgpack.unpackb(gz.read(), raw=False)
    state_dict = unpacked["nre_data"]["state_dict"]
    n_active = int(np.frombuffer(state_dict[".gaussians_nodes.gaussians.n_active_features"], dtype=np.int64)[0])
    return {
        "specular_coeffs": state_dict[".gaussians_nodes.gaussians.features_specular.shape"][1],
        "n_active_features": n_active,
        "radiance_sph_degree": unpacked["nre_data"]["config"]["layers"]["gaussians"]["particle"]["radiance_sph_degree"],
    }


class ResizeShCoefficientsTests(unittest.TestCase):
    def test_pad_appends_zeros_and_preserves_original(self):
        shN = np.random.rand(5, 3, 3).astype(np.float32)  # degree 1 -> 3 coeffs
        out = _resize_sh_coefficients(shN, target_sh_degree=3)  # -> 15 coeffs
        self.assertEqual(out.shape, (5, 15, 3))
        np.testing.assert_array_equal(out[:, :3, :], shN)  # original coefficients preserved
        np.testing.assert_array_equal(out[:, 3:, :], np.zeros((5, 12, 3), dtype=np.float32))  # padding is zero

    def test_truncate_keeps_leading_coefficients(self):
        shN = np.random.rand(5, 15, 3).astype(np.float32)  # degree 3 -> 15 coeffs
        out = _resize_sh_coefficients(shN, target_sh_degree=1)  # -> 3 coeffs
        self.assertEqual(out.shape, (5, 3, 3))
        np.testing.assert_array_equal(out, shN[:, :3, :])

    def test_identity_returns_input_unchanged(self):
        shN = np.random.rand(5, 8, 3).astype(np.float32)  # degree 2 -> already matches
        self.assertIs(_resize_sh_coefficients(shN, target_sh_degree=2), shN)

    def test_degree_zero_source(self):
        shN = np.zeros((5, 0, 3), dtype=np.float32)  # degree 0 -> no directional coeffs
        self.assertEqual(_resize_sh_coefficients(shN, target_sh_degree=0).shape, (5, 0, 3))
        self.assertEqual(_resize_sh_coefficients(shN, target_sh_degree=3).shape, (5, 15, 3))

    def test_negative_degree_raises(self):
        # A negative degree would otherwise silently drop coefficients via a negative-index slice.
        shN = np.zeros((5, 3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            _resize_sh_coefficients(shN, target_sh_degree=-1)

    def test_non_3d_input_raises(self):
        # A non-3D shN would otherwise fail with a cryptic tuple-unpack error.
        with self.assertRaises(ValueError):
            _resize_sh_coefficients(np.zeros((5, 3), dtype=np.float32), target_sh_degree=3)


class LegacyNurecShDegreeTests(unittest.TestCase):
    # Per-channel directional-coefficient counts the legacy NuRec importer accepts (SH degree 0 or 3).
    _SUPPORTED_SPECULAR_COEFFS = {0, 15}

    def _payload_info(self, model: GaussianSplat3d, **kwargs) -> dict:
        _, model_file = build_legacy_gaussians_payload(model, "scene", **kwargs)
        return _decode_nurec_payload(model_file.serialized)

    def test_auto_promotes_intermediate_degrees_to_supported_layout(self):
        # Degree 1 and 2 have an intermediate coefficient count that silently fails to import;
        # they must be promoted to the degree-3 layout (issue #124).
        for degree in (1, 2):
            with self.subTest(source_degree=degree):
                info = self._payload_info(_make_test_splats(sh_degree=degree))
                self.assertEqual(info["specular_coeffs"], 15)
                self.assertEqual(info["n_active_features"], 16)
                self.assertEqual(info["radiance_sph_degree"], 3)

    def test_auto_leaves_supported_degrees_unchanged(self):
        # Degree 0 and 3 already import correctly and must be exported unchanged.
        for degree, expected_coeffs, expected_n_active in ((0, 0, 1), (3, 15, 16)):
            with self.subTest(source_degree=degree):
                info = self._payload_info(_make_test_splats(sh_degree=degree))
                self.assertEqual(info["specular_coeffs"], expected_coeffs)
                self.assertEqual(info["n_active_features"], expected_n_active)

    def test_every_auto_export_uses_a_supported_layout(self):
        for degree in (0, 1, 2, 3):
            with self.subTest(source_degree=degree):
                info = self._payload_info(_make_test_splats(sh_degree=degree))
                self.assertIn(info["specular_coeffs"], self._SUPPORTED_SPECULAR_COEFFS)

    def test_explicit_target_pads_and_truncates(self):
        padded = self._payload_info(_make_test_splats(sh_degree=0), target_sh_degree=3)
        self.assertEqual(padded["specular_coeffs"], 15)
        self.assertEqual(padded["n_active_features"], 16)

        truncated = self._payload_info(_make_test_splats(sh_degree=3), target_sh_degree=0)
        self.assertEqual(truncated["specular_coeffs"], 0)
        self.assertEqual(truncated["n_active_features"], 1)

    def test_unsupported_explicit_target_is_rejected(self):
        # Explicit targets outside {0, 3} would produce a NuRec file the importer cannot load.
        model = _make_test_splats(sh_degree=3)
        for bad_degree in (-1, 1, 2, 4):
            with self.subTest(target_sh_degree=bad_degree):
                with self.assertRaises(ValueError):
                    build_legacy_gaussians_payload(model, "scene", target_sh_degree=bad_degree)

    def test_end_to_end_legacy_usdz_promotes_degree(self):
        # Full public path: a degree-2 model exported via export_splats_to_usd(legacy=True) must
        # produce a degree-3 NuRec payload inside the .usdz archive.
        model = _make_test_splats(sh_degree=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = export_splats_to_usd(model, Path(tmp_dir) / "scene", legacy=True, usdz=True)
            self.assertTrue(out_path.exists())
            with zipfile.ZipFile(out_path) as zf:
                nurec_name = next(name for name in zf.namelist() if name.endswith(".nurec"))
                info = _decode_nurec_payload(zf.read(nurec_name))
        self.assertEqual(info["specular_coeffs"], 15)


if __name__ == "__main__":
    unittest.main()
