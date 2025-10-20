.. code-block:: text

    usage: frgs show [-h] [SHOW OPTIONS] PATH

    Visualize a Gaussian splat radiance field in a saved PLY or checkpoint file. This will plot the
    splats in an interactive viewer shown in a browser window.

    # Example usage:

        # Visualize a Gaussian splat model saved in `model.ply`
        frgs show model.ply --viewer-port 8888

        # Visualize a Gaussian splat model saved in `model.pt`
        frgs show model.pt --viewer-port 8888

    ╭─ positional arguments ───────────────────────────────────────────────────────────────────────╮
    │ PATH                    Path to the input PLY or checkpoint file. Must end in .ply, .pt, or  │
    │                         .pth. (required)                                                     │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help              show this help message and exit                                      │
    │ -p INT, --viewer-port INT                                                                    │
    │                         The port to expose the viewer server on. (default: 8080)             │
    │ -ip STR, --viewer-ip-address STR                                                             │
    │                         The port to expose the viewer server on. (default: 127.0.0.1)        │
    │ -v, --verbose, --no-verbose                                                                  │
    │                         If True, then the viewer will log verbosely. (default: False)        │
    │ --device STR|DEVICE     Device to use for computation (default is "cuda"). (default: cuda)   │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯

