.. code-block:: text

    usage: frgs convert [-h] PATH PATH

    Convert a Gaussian Splat in one format to another. Currently the following conversions are
    supported:
    - PLY to USDZ
        - Checkpoint to USDZ
        - PLY to PLY (copy)
        - Checkpoint to PLY (export)

    Example usage:

        # Convert a PLY file to a USDZ file
        frgs frgs convert input.ply output.usdz

        # Convert a Checkpoint file to a USDZ file
        frgs frgs convert input.pt output.usdz

    ╭─ positional arguments ─────────────────────────────────────────────────────────────────────────╮
    │ PATH              Path to the input file. Must be a .ply file or Checkpoint (.pt or .pth)      │
    │                   file. (required)                                                             │
    │ PATH              Path to the output file. Must be a .ply file, Checkpoint (.pt or .pth) file, │
    │                   or .usdz file. (required)                                                    │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ──────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help        show this help message and exit                                              │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

