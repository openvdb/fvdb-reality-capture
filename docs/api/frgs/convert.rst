.. code-block:: text

    usage: frgs convert [-h] [CONVERT OPTIONS] PATH PATH

    Convert a Gaussian Splat in one format to another. Currently the following conversions are
    supported:
    - PLY to USD (.usdc single file, or .usdz archive)
    - Checkpoint to USD (.usdc single file, or .usdz archive)
    - PLY to PLY (copy)
    - Checkpoint to PLY (export)
    - PLY or checkpoint to USD with optional mesh and ecef2enu upright rotation

    Example usage:

        # Convert a PLY file to a single-file USDC
        frgs convert input.ply output.usdc

        # Convert a Checkpoint file to a single-file USDC
        frgs convert input.pt output.usdc

        # Convert a PLY file to a USDZ archive instead
        frgs convert input.ply output.usdc --usdz

        # Splats only, rotated for ecef2enu-normalized scenes
        frgs convert input.ply output.usdc --ecef2enu-rotation

        # Splats + mesh for Isaac Sim, rotated for ecef2enu-normalized scenes
        frgs convert input.ply output.usdc --mesh-path mesh.ply --ecef2enu-rotation

        # Legacy NuRec USDZ for Isaac Sim 5.x
        frgs convert input.ply output.usdc --legacy --usdz

        # Custom asset prim name (/World/my_asset) instead of the output file name
        frgs convert input.ply output.usdc --prim-path my_asset

    ╭─ positional arguments ───────────────────────────────────────────────────────╮
    │ PATH               Path to the input file. Must be a .ply file or Checkpoint │
    │                    (.pt or .pth) file. (required)                            │
    │ PATH               Path to the output file. Must be a .ply, .usdc, or .usdz  │
    │                    file. (required)                                          │
    ╰────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ────────────────────────────────────────────────────────────────────╮
    │ -h, --help         show this help message and exit                           │
    │ --mesh-path {None}|PATH                                                      │
    │                    USD export only. Optional mesh (PLY/OBJ) under            │
    │                    /World/<output_file_name>/mesh (shared asset xform).      │
    │                    (default: None)                                           │
    │ --ecef2enu-rotation, --no-ecef2enu-rotation                                  │
    │                    USD export only. Apply -90° X upright rotation on         │
    │                    /World/<output_file_name> for ecef2enu-normalized scenes. │
    │                    (default: False)                                          │
    │ --legacy, --no-legacy                                                        │
    │                    USD export only. Export legacy NuRec format              │
    │                    (UsdVol.Volume + .nurec) for Isaac Sim 5.x. Requires      │
    │                    --usdz. (default: False)                                  │
    │ --prim-path {None}|STR                                                       │
    │                    USD export only. Name of the asset prim placed under      │
    │                    /World (i.e. /World/<prim-path>). Defaults to the output   │
    │                    file name. (default: None)                                │
    │ --usdz, --no-usdz  USD export only. Package the export as a .usdz archive    │
    │                    instead of a single .usdc file. Implied if the output     │
    │                    path already ends in .usdz. (default: False)              │
    ╰────────────────────────────────────────────────────────────────────────────────╯
