# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import torch


def test_gaussian_splat_api_is_owned_by_reality_capture():
    import fvdb
    import fvdb_reality_capture

    public_symbols = (
        "GaussianSplat3d",
        "ProjectedGaussianSplats",
        "gaussian_render_jagged",
        "evaluate_spherical_harmonics",
        "gaussian_splat_to_view_data",
    )

    for symbol in public_symbols:
        assert hasattr(fvdb_reality_capture, symbol)
        assert not hasattr(fvdb, symbol)


def test_gaussian_splat_enums_are_owned_by_reality_capture_with_preserved_values():
    import fvdb
    import fvdb.viz
    import fvdb_reality_capture

    public_enums = (
        "RollingShutterType",
        "CameraModel",
        "ProjectionMethod",
    )

    for enum_name in public_enums:
        assert hasattr(fvdb_reality_capture, enum_name)
        assert not hasattr(fvdb, enum_name)

    assert not hasattr(fvdb_reality_capture, "ShOrderingMode")

    assert {member.name: member.value for member in fvdb_reality_capture.RollingShutterType} == {
        "NONE": 0,
        "VERTICAL": 1,
        "HORIZONTAL": 2,
    }
    assert {member.name: member.value for member in fvdb_reality_capture.CameraModel} == {
        "PINHOLE": 0,
        "OPENCV_RADTAN_5": 1,
        "OPENCV_RATIONAL_8": 2,
        "OPENCV_RADTAN_THIN_PRISM_9": 3,
        "OPENCV_THIN_PRISM_12": 4,
        "ORTHOGRAPHIC": 5,
    }
    assert {member.name: member.value for member in fvdb_reality_capture.ProjectionMethod} == {
        "AUTO": 0,
        "ANALYTIC": 1,
        "UNSCENTED": 2,
    }


def test_gaussian_splat_view_adapter_is_zero_copy_and_uses_the_core_contract():
    import fvdb.viz
    import fvdb_reality_capture as frc

    tensors = {
        "means": torch.randn(3, 3),
        "quats": torch.randn(3, 4),
        "log_scales": torch.randn(3, 3),
        "logit_opacities": torch.randn(3),
        "sh0": torch.randn(3, 1, 3),
        "shN": torch.randn(3, 8, 3),
    }
    model = frc.GaussianSplat3d.from_tensors(**tensors)

    view_data = frc.gaussian_splat_to_view_data(model)

    assert isinstance(view_data, fvdb.viz.GaussianSplatViewData)
    for name, tensor in tensors.items():
        assert getattr(view_data, name) is tensor
    assert view_data.sh_ordering == fvdb.viz.ShOrderingMode.RGB_RGB_RGB
    assert frc.gaussian_splat_to_view_data is frc.radiance_fields.gaussian_splat_to_view_data
    assert not hasattr(frc, "GaussianSplatViewData")
