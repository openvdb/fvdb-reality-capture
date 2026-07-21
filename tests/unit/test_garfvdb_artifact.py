# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import fvdb
import torch
from fvdb_reality_capture import GaussianSplat3d
from safetensors.torch import load_file, save_file

from fvdb_reality_capture.instance_segmentation import (
    ARTIFACT_SCHEMA_VERSION,
    GARfVDB,
    GARfVDBArtifactError,
    GARfVDBArtifactVersionError,
    GARfVDBConfig,
)
from fvdb_reality_capture.instance_segmentation.artifact import ARTIFACT_SCHEMA, MANIFEST_NAME
from fvdb_reality_capture.instance_segmentation.model import GARfVDBModel


def _make_gaussians(num_gaussians: int = 8) -> GaussianSplat3d:
    generator = torch.Generator().manual_seed(7)
    means = torch.rand((num_gaussians, 3), generator=generator)
    quats = torch.rand((num_gaussians, 4), generator=generator)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    log_scales = torch.full((num_gaussians, 3), -3.0)
    logit_opacities = torch.zeros(num_gaussians)
    sh0 = torch.rand((num_gaussians, 1, 3), generator=generator)
    shN = torch.zeros((num_gaussians, 0, 3))
    return GaussianSplat3d.from_tensors(means, quats, log_scales, logit_opacities, sh0, shN)


def _make_model(device: str) -> GARfVDBModel:
    gaussians = _make_gaussians().to(device)
    config = GARfVDBConfig(
        num_grids=4,
        grid_feature_dim=2,
        mlp_hidden_dim=8,
        mlp_num_layers=1,
        mlp_output_dim=4,
    )
    return GARfVDBModel(
        gaussians,
        torch.tensor([0.05, 0.1, 0.2, 0.4], device=device),
        model_config=config,
        device=device,
    )


def _make_product() -> GARfVDB:
    model = _make_model("cuda:0")
    return GARfVDB(model, reconstruction_metadata={"normalization_transform": torch.eye(4)})


def _refresh_payload_checksum(bundle_path: Path, payload_name: str) -> None:
    payload_path = bundle_path / payload_name
    manifest_path = bundle_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"][payload_name] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


class GARfVDBNanoVDBTests(unittest.TestCase):
    def test_encoder_feature_round_trip_on_cpu(self):
        grid = fvdb.GridBatch.from_dense(2, [2, 2, 2], device="cpu")
        features = grid.jagged_like(torch.arange(grid.total_voxels * 8).reshape(grid.total_voxels, 8).float())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.nvdb"
            grid.save_nanovdb(str(path), data=features, names=["encoder_00", "encoder_01"], compressed=True)
            loaded_grid, loaded_features, names = fvdb.functional.load_nanovdb(str(path), device="cpu")
        self.assertEqual(names, ["encoder_00", "encoder_01"])
        self.assertEqual(loaded_grid.num_voxels.tolist(), grid.num_voxels.tolist())
        torch.testing.assert_close(loaded_grid.ijk.jdata, grid.ijk.jdata)
        torch.testing.assert_close(loaded_grid.voxel_sizes, grid.voxel_sizes)
        torch.testing.assert_close(loaded_grid.origins, grid.origins)
        self.assertEqual(loaded_features.jdata.shape, features.jdata.shape)
        self.assertEqual(loaded_features.jdata.dtype, features.jdata.dtype)
        torch.testing.assert_close(loaded_features.jdata, features.jdata)

    def test_gaussian_affinities_use_raw_encoder_features(self):
        model = _make_model("cpu")
        encoder_features = model._sample_encoder_grids_at_gaussians()
        normalized_features = encoder_features / (encoder_features.norm(dim=-1, keepdim=True) + 1e-6)
        expected = model.get_mlp_output(normalized_features, 0.1)

        torch.testing.assert_close(model.get_gaussian_affinity_output(0.1), expected)

    def test_nanovdb_and_safetensors_reconstruct_model(self):
        model = _make_model("cpu")
        expected = model.get_gaussian_affinity_output(0.1)
        names = [f"encoder_{index:02d}" for index in range(model.encoder_gridbatch.grid_count)]
        with tempfile.TemporaryDirectory() as directory:
            encoder_path = Path(directory) / "encoder.nvdb"
            network_path = Path(directory) / "network.safetensors"
            model.encoder_gridbatch.save_nanovdb(
                str(encoder_path), data=model.enc_features, names=names, compressed=True
            )
            expected_network = model.network_state_dict()
            save_file(expected_network, network_path)
            loaded_network = load_file(network_path, device="cpu")
            self.assertEqual(loaded_network.keys(), expected_network.keys())
            for key, expected_tensor in expected_network.items():
                torch.testing.assert_close(loaded_network[key], expected_tensor)
            grid, features, loaded_names = fvdb.functional.load_nanovdb(str(encoder_path), device="cpu")
            restored = GARfVDBModel.from_artifact_components(
                gs_model=model.gs_model,
                model_config=model.model_config,
                encoder_gridbatch=grid,
                encoder_features=features.jdata,
                network_state=loaded_network,
                device="cpu",
            )
        self.assertEqual(loaded_names, names)
        torch.testing.assert_close(restored.get_gaussian_affinity_output(0.1), expected)


class GARfVDBSchemaVersionTests(unittest.TestCase):
    def _write_manifest(self, bundle_path: Path, manifest: dict) -> None:
        bundle_path.mkdir()
        (bundle_path / MANIFEST_NAME).write_text(json.dumps(manifest))

    def test_newer_schema_version_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.garfvdb"
            self._write_manifest(
                path,
                {
                    "schema": ARTIFACT_SCHEMA,
                    "schema_version": ARTIFACT_SCHEMA_VERSION + 1,
                },
            )
            with self.assertRaisesRegex(GARfVDBArtifactVersionError, "newer version"):
                GARfVDB.load(path, device="cpu")

    def test_unversioned_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unversioned.garfvdb"
            self._write_manifest(path, {"schema": ARTIFACT_SCHEMA})
            with self.assertRaisesRegex(GARfVDBArtifactVersionError, "schema_version"):
                GARfVDB.load(path, device="cpu")

    def test_generic_version_field_is_not_treated_as_schema_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ambiguous.garfvdb"
            self._write_manifest(path, {"schema": ARTIFACT_SCHEMA, "version": 1})
            with self.assertRaisesRegex(GARfVDBArtifactVersionError, "schema_version"):
                GARfVDB.load(path, device="cpu")


@unittest.skipUnless(torch.cuda.is_available(), "GARfVDB artifact round trips require CUDA Gaussian PLY I/O")
class GARfVDBArtifactTests(unittest.TestCase):
    def test_round_trip_uses_nanovdb_and_safetensors(self):
        product = _make_product()
        expected_affinities = product.gaussian_affinities(0.1)
        projection = torch.tensor(
            [[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]],
            device="cuda:0",
        )
        expected_render, expected_alpha = product.render_features(
            torch.eye(4, device="cuda:0"), projection, (16, 16), 0.1
        )
        expected_grid = product.encoder_grids
        expected_features = product.model.encoder_gridbatch_features_data.detach()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.garfvdb"
            product.save(path)

            self.assertTrue((path / "encoder.nvdb").is_file())
            self.assertTrue((path / "network.safetensors").is_file())
            self.assertTrue((path / "gaussians.ply").is_file())
            self.assertFalse((path / "model.pt").exists())
            manifest = json.loads((path / MANIFEST_NAME).read_text())
            self.assertEqual(manifest["schema_version"], ARTIFACT_SCHEMA_VERSION)
            self.assertNotIn("version", manifest)

            relocated_path = Path(directory) / "relocated.garfvdb"
            path.rename(relocated_path)
            loaded = GARfVDB.load(relocated_path, device="cuda:0")
            self.assertEqual(loaded.encoder_grids.grid_count, expected_grid.grid_count)
            self.assertEqual(loaded.encoder_grids.num_voxels.tolist(), expected_grid.num_voxels.tolist())
            torch.testing.assert_close(loaded.encoder_grids.voxel_sizes, expected_grid.voxel_sizes)
            torch.testing.assert_close(loaded.encoder_grids.origins, expected_grid.origins)
            torch.testing.assert_close(loaded.model.encoder_gridbatch_features_data, expected_features)
            torch.testing.assert_close(loaded.gaussian_affinities(0.1), expected_affinities)
            loaded_render, loaded_alpha = loaded.render_features(torch.eye(4, device="cuda:0"), projection, (16, 16), 0.1)
            torch.testing.assert_close(loaded_render, expected_render)
            torch.testing.assert_close(loaded_alpha, expected_alpha)

    def test_payload_checksum_is_validated(self):
        product = _make_product()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.garfvdb"
            product.save(path)
            with (path / "network.safetensors").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(GARfVDBArtifactError, "checksum"):
                GARfVDB.load(path, device="cuda:0")

    def test_grid_names_are_validated(self):
        product = _make_product()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.garfvdb"
            product.save(path)
            manifest_path = path / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            manifest["encoder"]["grid_names"][0] = "wrong"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(GARfVDBArtifactError, "grid names"):
                GARfVDB.load(path, device="cuda:0")

    def test_reordered_nanovdb_grid_names_are_rejected(self):
        product = _make_product()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.garfvdb"
            product.save(path)
            encoder_path = path / "encoder.nvdb"
            grid, features, names = fvdb.functional.load_nanovdb(str(encoder_path), device="cuda:0")
            encoder_path.unlink()
            grid.save_nanovdb(str(encoder_path), data=features, names=list(reversed(names)), compressed=True)
            _refresh_payload_checksum(path, "encoder.nvdb")
            with self.assertRaisesRegex(GARfVDBArtifactError, "grid names"):
                GARfVDB.load(path, device="cuda:0")

    def test_missing_nanovdb_grid_is_rejected(self):
        product = _make_product()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.garfvdb"
            product.save(path)
            # Simulate an encoder payload that is missing a grid: the manifest expects one more grid than
            # the NanoVDB file actually contains, which the loader's grid-count check must reject. (fvdb's
            # grid-slice save is unstable under the current build, so we drive the same validation via the
            # manifest instead of re-saving a subset of grids.)
            manifest_path = path / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            manifest["encoder"]["grid_count"] += 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(GARfVDBArtifactError):
                GARfVDB.load(path, device="cuda:0")

    def test_incompatible_nanovdb_transform_is_rejected(self):
        product = _make_product()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.garfvdb"
            product.save(path)
            encoder_path = path / "encoder.nvdb"
            grid, features, names = fvdb.functional.load_nanovdb(str(encoder_path), device="cuda:0")
            incompatible_grid = fvdb.GridBatch.from_ijk(
                grid.ijk,
                voxel_sizes=grid.voxel_sizes * 2,
                origins=grid.origins,
            )
            encoder_path.unlink()
            incompatible_grid.save_nanovdb(
                str(encoder_path),
                data=incompatible_grid.jagged_like(features.jdata),
                names=names,
                compressed=True,
            )
            _refresh_payload_checksum(path, "encoder.nvdb")
            with self.assertRaisesRegex(GARfVDBArtifactError, "transforms"):
                GARfVDB.load(path, device="cuda:0")

    def test_artifact_requires_grid_backed_model(self):
        gaussians = _make_gaussians().to("cuda:0")
        config = GARfVDBConfig(use_grid=False, num_grids=2, grid_feature_dim=2, mlp_hidden_dim=4, mlp_num_layers=1)
        model = GARfVDBModel(gaussians, torch.tensor([0.1, 0.2], device="cuda:0"), config, device="cuda:0")
        with self.assertRaisesRegex(ValueError, "use_grid=True"):
            GARfVDB(model)


if __name__ == "__main__":
    unittest.main()
