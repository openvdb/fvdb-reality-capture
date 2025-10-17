.. code-block:: text

    usage: frgs evaluate [-h] [EVALUATE OPTIONS] PATH

    Evaluate a Gaussian Splat reconstruction on the validation set. You can change the validation
    split used for evaluation using `--use-every-n-as-val` argument. For example,
    `--use-every-n-as-val 10` will use every 10th image in the dataset as a validation image. If you
    do not provide this argument, the validation split provided in the dataset (if any) will be used.
    If the dataset does not provide a validation split, all images will be used for evaluation.

    This will render each image in the validation set, compute statistics (PSNR, SSIM, LPIPS), and
    save the rendered images and ground truth validation images to disk.

    By default results will be saved to a directory named "eval" in the same directory as the
    checkpoint.

    Example usage:

        # Evaluate a checkpoint and save results to the default log path
        frgs evaluate checkpoint.pt

        # Evaluate a checkpoint on a new dataset split
        frgs evaluate checkpoint.pt --use-every-n-as-val 10

        # Evaluate a checkpoint and save results to a custom log path
        frgs evaluate checkpoint.pt --log-path ./eval_results

        # Evaluate a checkpoint but don't write out rendered images
        frgs evaluate checkpoint.pt --save-images False

    ╭─ positional arguments ─────────────────────────────────────────────────────────────────────────╮
    │ PATH                                                                                           │
    │     Path to the checkpoint file containing the Gaussian Splat model. (required)                │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ──────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help                                                                                     │
    │     show this help message and exit                                                            │
    │ -l {None}|PATH, --log-path {None}|PATH                                                         │
    │     Path to save the evaluation results. If not provided, results will be saved in a           │
    │     subdirectory of the checkpoint directory named "eval". (default: None)                     │
    │ -vn {None}|INT, --use-every-n-as-val {None}|INT                                                │
    │     Use every n-th image as a validation image. If not set, will use the validation split      │
    │     provided in the dataset. If the dataset does not provide a validation split, will use all  │
    │     images for evaluation. (default: None)                                                     │
    │ -s, --save-images, --no-save-images                                                            │
    │     Whether to save the rendered images. Defaults to True. (default: True)                     │
    │ --device STR|DEVICE                                                                            │
    │     Device to use for computation. Defaults to "cuda". (default: cuda)                         │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

