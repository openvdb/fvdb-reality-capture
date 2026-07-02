.. code-block:: text

    usage: frgs convert [-h] PATH PATH

    Convert a Gaussian Splat in one format to another. Currently the following conversions are
    supported:
    - PLY to USDZ
    - Checkpoint to USDZ
    - PLY to PLY (copy)
    - Checkpoint to PLY (export)
    - PLY or checkpoint to USDZ with optional mesh and ecef2enu upright rotation

    Example usage:

        # Convert a PLY file to a USDZ file
        frgs convert input.ply output.usdz

        # Convert a Checkpoint file to a USDZ file
        frgs convert input.pt output.usdz

        # Splats + mesh for Isaac Sim, rotated for ecef2enu-normalized scenes
        frgs convert input.ply output.usdz --mesh-path mesh.ply --ecef2enu-rotation

        # Legacy NuRec USDZ for Isaac Sim 5.x
        frgs convert input.ply output.usdz --legacy

        # Custom asset prim name (/World/my_asset) instead of the output file name
        frgs convert input.ply output.usdz --prim-path my_asset

    ╭─ positional arguments ─────────────────────────────────────────────────────────────────────────╮
    │ PATH              Path to the input file. Must be a .ply file or Checkpoint (.pt or .pth)      │
    │                   file. (required)                                                             │
    │ PATH              Path to the output file. Must be a .ply file or .usdz file. (required)     │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ──────────────────────────────────────────────────────────────────────────────────────╮
    │ --mesh-path PATH       USDZ only. Mesh under /World/<output_file_name>/mesh (shared asset xform)      │
    │ --ecef2enu-rotation    USDZ only. -90° X upright rotation on /World/<output_file_name>                 │
    │ --legacy               USDZ only. NuRec format for Isaac Sim 5.x                            │
    │ --prim-path STR        USDZ only. Asset prim name under /World; defaults to output file name  │
    │ -h, --help             show this help message and exit                                       │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
