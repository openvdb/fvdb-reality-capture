# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0


def test_gaussian_splat_api_is_owned_by_reality_capture():
    import fvdb
    import fvdb_reality_capture

    public_symbols = (
        "GaussianSplat3d",
        "ProjectedGaussianSplats",
        "gaussian_render_jagged",
        "evaluate_spherical_harmonics",
    )

    for symbol in public_symbols:
        assert hasattr(fvdb_reality_capture, symbol)
        assert not hasattr(fvdb, symbol)


def test_gaussian_splat_enums_are_owned_by_reality_capture_with_preserved_values():
    import fvdb
    import fvdb.viz
    import fvdb_reality_capture

    public_enums = (
        "ShOrderingMode",
        "RollingShutterType",
        "CameraModel",
        "ProjectionMethod",
    )

    for enum_name in public_enums:
        assert hasattr(fvdb_reality_capture, enum_name)
        assert not hasattr(fvdb, enum_name)

    assert not hasattr(fvdb.viz, "ShOrderingMode")

    assert {member.name: member.value for member in fvdb_reality_capture.ShOrderingMode} == {
        "RGB_RGB_RGB": "rgb_rgb_rgb",
        "RRR_GGG_BBB": "rrr_ggg_bbb",
    }
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
