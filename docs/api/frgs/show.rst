.. code-block:: text

    usage: frgs show [-h] [SHOW OPTIONS] PATH

    Visualize a saved Reality Capture product. Supported inputs are Gaussian PLY/checkpoint files
    and portable GARfVDB bundle directories.

    # Example usage:

        # Visualize a Gaussian splat model saved in `model.ply`
        frgs show model.ply --viewer-port 8888

        # Visualize a Gaussian splat model saved in `model.pt`
        frgs show model.pt --viewer-port 8888

    ╭─ positional arguments ───────────────────────────────────────────────────────────────────────╮
    │ PATH                    Path to a PLY, checkpoint, or .garfvdb bundle. (required)            │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help              show this help message and exit                                      │
    │ -p INT, --viewer-port INT                                                                    │
    │                         The port to expose the viewer server on. (default: 8080)             │
    │ -ip STR, --viewer-ip-address STR                                                             │
    │                         The port to expose the viewer server on. (default: 127.0.0.1)        │
    │ -v, --verbose, --no-verbose                                                                  │
    │                         If True, then the viewer will log verbosely. (default: False)        │
    │ --device STR|DEVICE     Device to use for computation (default is "cuda:0"). (default: cuda:0) │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
