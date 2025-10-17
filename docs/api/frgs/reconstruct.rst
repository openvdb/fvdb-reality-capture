.. code-block:: text

    usage: frgs reconstruct [-h] [RECONSTRUCT OPTIONS] PATH

    Reconstruct a Gaussian Splat Radiance Field from a dataset of posed images, and save the result as
    a PLY or USDZ file. Example usage:

        # Reconstruct a Gaussian splat radiance field from a Colmap dataset
        frgs reconstruct ./colmap_dataset -o ./output.ply

        # Reconstruct a Gaussian splat radiance field from a dataset of e57 files
        frgs reconstruct ./simple_directory_dataset --dataset-type e57 --out-path ./output.usdz

    ╭─ positional arguments ─────────────────────────────────────────────────────────────────────────╮
    │ PATH                                                                                           │
    │     Path to the dataset. For "colmap" datasets, this should be the directory containing the    │
    │     `images` and `sparse` subdirectories. For "simple_directory" datasets, this should be the  │
    │     directory containing the images and a `cameras.txt` file. (required)                       │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ options ──────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help                                                                                     │
    │     show this help message and exit                                                            │
    │ -o PATH, --out-path PATH                                                                       │
    │     Path to save the output PLY file. Defaults to `out.ply` in the current working directory.  │
    │     Path must end in .ply or .usdz. (default: out.ply)                                         │
    │ -n {None}|STR, --run-name {None}|STR                                                           │
    │     Name of the run. If None, a name will be generated based on the current date and time.     │
    │     (default: None)                                                                            │
    │ -dt {colmap,simple_directory,e57}, --dataset-type {colmap,simple_directory,e57}                │
    │     Type of dataset to load. (default: colmap)                                                 │
    │ -vn INT, --use-every-n-as-val INT                                                              │
    │     Use every n-th image as a validation image. If -1, do not use a validation set. (default:  │
    │     -1)                                                                                        │
    │ -uv FLOAT, --update-viz-every FLOAT                                                            │
    │     How frequently (in epochs) to update the viewer during reconstruction. An epoch is one     │
    │     full pass through the dataset. If -1, do not visualize. (default: -1.0)                    │
    │ -d STR|DEVICE, --device STR|DEVICE                                                             │
    │     Which device to use for reconstruction. Must be a cuda device. You can pass in a specific  │
    │     device index via cuda:N where N is the device index, or "cuda" to use the default cuda     │
    │     device. CPU is not supported. Default is "cuda". (default: cuda)                           │
    │ -v, --verbose, --no-verbose                                                                    │
    │     If set, show verbose debug messages. (default: False)                                      │
    │ -nc INT INT INT, --nchunks INT INT INT                                                         │
    │     Configuration to split the dataset into chunks for reconstruction. If set to (1, 1, 1),    │
    │     the dataset will not be chunked. (default: 1 1 1)                                          │
    │ -nco FLOAT, --chunk-overlap-pct FLOAT                                                          │
    │     Percentage of overlap between chunks if reconstructing in chunks. Must be in [0, 1]. Only  │
    │     used if nchunks is not (1, 1, 1). Default is 0.1 (10% overlap). (default: 0.1)             │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ cfg options ──────────────────────────────────────────────────────────────────────────────────╮
    │ Configuration parameters for the Gaussian splat reconstruction.                                │
    │ ────────────────────────────────────────────────────────────────────────────────────────────── │
    │ --cfg.seed INT                                                                                 │
    │     A random seed for reproducibility.                                                         │
    │                                                                                                │
    │                                                                                                │
    │     Default: 42 (the meaning of life, the universe, and everything). (default: 42)             │
    │ --cfg.max-epochs INT                                                                           │
    │     The maximum number of optimization epochs, *i.e.*, the number of times each image in the   │
    │     dataset will be visited.                                                                   │
    │                                                                                                │
    │                                                                                                │
    │     An epoch is defined as one full pass through the dataset. If you have a dataset with 100   │
    │     images and a batch                                                                         │
    │     size of 10, then one epoch corresponds to 10 steps.                                        │
    │                                                                                                │
    │                                                                                                │
    │     Default: 200 (default: 200)                                                                │
    │ --cfg.max-steps {None}|INT                                                                     │
    │     The maximum number of optimization steps. If set, this overrides the number of steps       │
    │     calculated from `max_epochs` and the dataset size.                                         │
    │                                                                                                │
    │                                                                                                │
    │     You shouldn't use this parameter unless you have a specific reason to do so.               │
    │                                                                                                │
    │                                                                                                │
    │     Default: None (default: None)                                                              │
    │ --cfg.eval-at-percent [INT [INT ...]]                                                          │
    │     Percentage of the total optimization epochs at which to perform evaluation on the          │
    │     validation set.                                                                            │
    │                                                                                                │
    │                                                                                                │
    │     For example, if `eval_at_percent` is set to `[10, 50, 100]` and `max_epochs` is set to     │
    │     `200`, then evaluation will be                                                             │
    │     performed after 20, 100, and 200 epochs.                                                   │
    │                                                                                                │
    │                                                                                                │
    │     Default: [10, 20, 30, 40, 50, 75, 100] (default: 10 20 30 40 50 75 100)                    │
    │ --cfg.save-at-percent [INT [INT ...]]                                                          │
    │     Percentage of the total optimization epochs at which to save model checkpoints.            │
    │                                                                                                │
    │                                                                                                │
    │     For example, if `save_at_percent` is set to `[50, 100]` and `max_epochs` is set to `200`,  │
    │     then checkpoints will be saved after 100 and 200 epochs.                                   │
    │                                                                                                │
    │                                                                                                │
    │     Default: [20, 100] (default: 20 100)                                                       │
    │ --cfg.batch-size INT                                                                           │
    │     Batch size for optimization. Each step of optimization will compute losses on              │
    │     :obj:`batch_size` images. Note that                                                        │
    │     learning rates are scaled automatically based on the batch size.                           │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``1`` (default: 1)                                                                │
    │ --cfg.crops-per-image INT                                                                      │
    │     Number of crops to use per image during reconstruction. If you're using very large images, │
    │     you can set this to a value greater than 1                                                 │
    │     to run the forward pass on crops and accumulate gradients. This can help reduce memory     │
    │     usage.                                                                                     │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``1`` (no cropping, use full images). (default: 1)                                │
    │ --cfg.sh-degree INT                                                                            │
    │     Maximum degree of spherical harmonics to use for each Gaussian's view-dependent color.     │
    │     Higher degrees allow for more complex view-dependent effects, but increase memory usage    │
    │     and computation time.                                                                      │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``3`` (default: 3)                                                                │
    │ --cfg.increase-sh-degree-every-epoch INT                                                       │
    │     When reconstructing a Gaussian splat radiance field, we start by only optimizing the       │
    │     diffuse (degree 0) spherical harmonics coefficients                                        │
    │     per Gaussian, and progressively increase the degree of spherical harmonics used every      │
    │     :obj:`increase_sh_degree_every_epoch` epochs                                               │
    │     until we reach :obj:`sh_degree`. This helps stabilize optimization in the early stages of  │
    │     optimization.                                                                              │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``5`` (default: 5)                                                                │
    │ --cfg.initial-opacity FLOAT                                                                    │
    │     Initial opacity of each Gaussian. This is the alpha value used when rendering the          │
    │     Gaussians at the start of optimization.                                                    │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0.1`` (default: 0.1)                                                            │
    │ --cfg.initial-covariance-scale FLOAT                                                           │
    │     Initial scale of each Gaussian. This controls the initial size of the Gaussians in the     │
    │     scene.                                                                                     │
    │     Each Gaussian's covariance matrix will be initialized to a diagonal matrix with this value │
    │     on the diagonal.                                                                           │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``1.0`` (default: 1.0)                                                            │
    │ --cfg.ssim-lambda FLOAT                                                                        │
    │     Weight for SSIM loss. Reconstruction aims to minimize                                      │
    │     the `Structural Similarity Index Measure (SSIM)                                            │
    │     <https://en.wikipedia.org/wiki/Structural_similarity_index_measure>`_                      │
    │     between rendered images with the radiance field and ground truth images. This weight       │
    │     applies to the SSIM loss term.                                                             │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0.2`` (default: 0.2)                                                            │
    │ --cfg.lpips-net {vgg,alex}                                                                     │
    │     During evaluation, we compute the `Learned Perceptual Image Patch Similarity (LPIPS)       │
    │     <https://arxiv.org/abs/1801.03924>`_ metric                                                │
    │     as a measure of quality of the reconstruction. This parameter controls which network       │
    │     architecture is used for the LPIPS metric.                                                 │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``"alex"`` meaning the `AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_         │
    │     architecture. (default: alex)                                                              │
    │ --cfg.opacity-reg FLOAT                                                                        │
    │     Weight for opacity regularization loss :math:`L_{opacity} = \frac{1}{N} \sum_i             │
    │     |opacity_i|`.                                                                              │
    │                                                                                                │
    │                                                                                                │
    │     If set to a value greater than 0, this will encourage the opacities of the Gaussians to be │
    │     small.                                                                                     │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0.0`` (no opacity regularization). (default: 0.0)                               │
    │ --cfg.scale-reg FLOAT                                                                          │
    │     Weight for scale regularization loss :math:`L_{scale} = \frac{1}{N} \sum_i |scale_i|`.     │
    │                                                                                                │
    │                                                                                                │
    │     If set to a value greater than 0, this will encourage the scales of the Gaussians to be    │
    │     small.                                                                                     │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0.0`` (no scale regularization). (default: 0.0)                                 │
    │ --cfg.random-bkgd, --cfg.no-random-bkgd                                                        │
    │     Whether to render images with the radiance field against a background of random values     │
    │     during optimization.                                                                       │
    │     This discourages the model from using transparency to minimize loss.                       │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``False`` (default: False)                                                        │
    │ --cfg.refine-start-epoch INT                                                                   │
    │     At which epoch to start refining the Gaussians by inserting and deleting Gaussians based   │
    │     on their contribution to the optimization.                                                 │
    │     *e.g.* If this value is 3, the first refinement will occur at the start of epoch 3.        │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``3`` (default: 3)                                                                │
    │ --cfg.refine-stop-epoch INT                                                                    │
    │     At which epoch to stop refining the Gaussians by inserting and deleting Gaussians based on │
    │     their contribution to the optimization.                                                    │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``100`` (default: 100)                                                            │
    │ --cfg.refine-every-epoch FLOAT                                                                 │
    │     How often to refine Gaussians during optimization, in terms of epochs.                     │
    │     For example, a value of 0.65 means refinement occurs approximately every 0.65 epochs.      │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0.65`` (default: 0.65)                                                          │
    │ --cfg.ignore-masks, --cfg.no-ignore-masks                                                      │
    │     If set to ``True``, then ignore any masks in the data and treat all pixels as valid during │
    │     optimization.                                                                              │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``False`` (default: False)                                                        │
    │ --cfg.remove-gaussians-outside-scene-bbox, --cfg.no-remove-gaussians-outside-scene-bbox        │
    │     If set to ``True``, then Gaussians that fall outside the scene bounding box will be        │
    │     removed during refinement.                                                                 │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``False`` (default: False)                                                        │
    │ --cfg.optimize-camera-poses, --cfg.no-optimize-camera-poses                                    │
    │     If set to ``True``, optimize camera poses during reconstruction. This can help improve the │
    │     quality of the reconstruction if the initial poses are not accurate.                       │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``True`` (default: True)                                                          │
    │ --cfg.pose-opt-lr FLOAT                                                                        │
    │     Learning rate for camera pose optimization.                                                │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``1e-5`` (default: 1e-05)                                                         │
    │ --cfg.pose-opt-reg FLOAT                                                                       │
    │     Weight for regularization of camera pose optimization. This encourages small changes to    │
    │     the initial camera poses.                                                                  │
    │                                                                                                │
    │                                                                                                │
    │     The pose regularization loss is defined as :math:`L_{pose}` = \frac{1}{M} \sum_j ||\Delta  │
    │     R_j||^2 + ||\Delta t_j||^2`,                                                               │
    │     *i.e.* the Frobenius norm of the change in rotation and translation for each of the ``M``  │
    │     camera poses in the dataset.                                                               │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``1e-6`` (default: 1e-06)                                                         │
    │ --cfg.pose-opt-lr-decay FLOAT                                                                  │
    │     Learning rate decay factor for camera pose optimization (will decay to this fraction of    │
    │     initial lr).                                                                               │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``1.0`` (no decay). (default: 1.0)                                                │
    │ --cfg.pose-opt-start-epoch INT                                                                 │
    │     At which epoch to start optimizing camera poses.                                           │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0`` (start from beginning of optimization). (default: 0)                        │
    │ --cfg.pose-opt-stop-epoch INT                                                                  │
    │     At which epoch to stop optimizing camera poses.                                            │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``max_epochs`` (optimize poses for the entire duration of optimization).          │
    │     (default: 200)                                                                             │
    │ --cfg.pose-opt-init-std FLOAT                                                                  │
    │     Standard deviation for the normal distribution used to initialize the embeddings for       │
    │     camera pose optimization.                                                                  │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``1e-4`` (default: 0.0001)                                                        │
    │ --cfg.near-plane FLOAT                                                                         │
    │     Near plane clipping distance when rendering the Gaussians.                                 │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0.01`` (default: 0.01)                                                          │
    │ --cfg.far-plane FLOAT                                                                          │
    │     Far plane clipping distance when rendering the Gaussians.                                  │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``1e10`` (default: 10000000000.0)                                                 │
    │ --cfg.min-radius-2d FLOAT                                                                      │
    │     Minimum screen space radius (in pixels) below which Gaussians are ignored after            │
    │     projection.                                                                                │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0.0`` (default: 0.0)                                                            │
    │ --cfg.eps-2d FLOAT                                                                             │
    │     Amount of padding (in pixels) to add to the screen space bounding box of each Gaussian     │
    │     when determining which pixels it affects.                                                  │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``0.3`` (default: 0.3)                                                            │
    │ --cfg.antialias, --cfg.no-antialias                                                            │
    │     Whether to use anti-aliasing when rendering the Gaussians.                                 │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``False`` (default: False)                                                        │
    │ --cfg.tile-size INT                                                                            │
    │     Tile size (in pixels) to use when rendering the Gaussians.                                 │
    │     You should generally leave this at the default value unless you have a specific reason to  │
    │     change it.                                                                                 │
    │                                                                                                │
    │                                                                                                │
    │     Default: ``16`` (default: 16)                                                              │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ tx options ───────────────────────────────────────────────────────────────────────────────────╮
    │ Configuration for the transforms to apply to the scene before reconstruction.                  │
    │ ────────────────────────────────────────────────────────────────────────────────────────────── │
    │ --tx.image-downsample-factor INT                                                               │
    │     Downsample images by this factor (default: 4)                                              │
    │ --tx.rescale-jpeg-quality INT                                                                  │
    │     JPEG quality to use when resaving images after downsampling (default: 95)                  │
    │ --tx.points-percentile-filter FLOAT                                                            │
    │     Percentile of points to filter out based on their distance from the median point (default: │
    │     0.0)                                                                                       │
    │ --tx.normalization-type {none,pca,ecef2enu,similarity}                                         │
    │     Type of normalization to apply to the scene (default: pca)                                 │
    │ --tx.crop-bbox {None}|{FLOAT FLOAT FLOAT FLOAT FLOAT FLOAT}                                    │
    │     Optional bounding box (in the normalized space) to crop the scene to (xmin, xmax, ymin,    │
    │     ymax, zmin, zmax) (default: None)                                                          │
    │ --tx.crop-to-points, --tx.no-crop-to-points                                                    │
    │     Whether to crop the scene to the bounding box or not (default: False)                      │
    │ --tx.min-points-per-image INT                                                                  │
    │     Minimum number of 3D points that must be visible in an image for it to be included in the  │
    │     optimization (default: 5)                                                                  │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ opt options ──────────────────────────────────────────────────────────────────────────────────╮
    │ Configuration for the optimizer used to reconstruct the Gaussian splat radiance field.         │
    │ ────────────────────────────────────────────────────────────────────────────────────────────── │
    │ --opt.max-gaussians INT                                                                        │
    │     The maximum number of Gaussians to allow in the model. If -1, no limit. (default: -1)      │
    │ --opt.insertion-grad-2d-threshold-mode                                                         │
    │ {CONSTANT,PERCENTILE_FIRST_ITERATION,PERCENTILE_EVERY_ITERATION}                               │
    │     Whether to use a fixed threshold for :obj:`insertion_grad_2d_threshold` (constant), a      │
    │     value computed as a percentile of                                                          │
    │     the distribution of screen space mean gradients on the first iteration, or a percentile    │
    │     value                                                                                      │
    │     computed at each refinement step.                                                          │
    │                                                                                                │
    │                                                                                                │
    │     See :class:`InsertionGrad2dThresholdMode` for details on the available modes. (default:    │
    │     CONSTANT)                                                                                  │
    │ --opt.deletion-opacity-threshold FLOAT                                                         │
    │     If a Gaussian's opacity drops below this value, delete it during refinement. (default:     │
    │     0.005)                                                                                     │
    │ --opt.deletion-scale-3d-threshold FLOAT                                                        │
    │     If a Gaussian's 3d scale is above this value, then delete it during refinement. (default:  │
    │     0.1)                                                                                       │
    │ --opt.deletion-scale-2d-threshold FLOAT                                                        │
    │     If the maximum projected size of a Gaussian between refinement steps exceeds this value    │
    │     then delete it during                                                                      │
    │     refinement.                                                                                │
    │                                                                                                │
    │                                                                                                │
    │     .. note:: This parameter is only used if set                                               │
    │     :obj:`use_screen_space_scales_for_refinement_until` is greater than 0. (default: 0.15)     │
    │ --opt.insertion-grad-2d-threshold FLOAT                                                        │
    │     Threshold value on the accumulated norm of projected mean gradients between refinement     │
    │     steps to                                                                                   │
    │     determine whether a Gaussian has high error and is a candidate for duplication or          │
    │     splitting.                                                                                 │
    │                                                                                                │
    │                                                                                                │
    │     .. note:: If :obj:`insertion_grad_2d_threshold_mode` is                                    │
    │     :obj:`InsertionGrad2dThresholdMode.CONSTANT`, then this value                              │
    │               is used directly as the threshold, and **must be positive**.                     │
    │                                                                                                │
    │                                                                                                │
    │     .. note:: If :obj:`insertion_grad_2d_threshold_mode` is                                    │
    │     :obj:`InsertionGrad2dThresholdMode.PERCENTILE_FIRST_ITERATION`                             │
    │               or :obj:`InsertionGrad2dThresholdMode.PERCENTILE_EVERY_ITERATION`, then this     │
    │     value must be in the                                                                       │
    │               range ``(0.0, 1.0)`` (exclusive). (default: 0.0002)                              │
    │ --opt.insertion-scale-3d-threshold FLOAT                                                       │
    │     Duplicate high-error (determined by :obj:`insertion_grad_2d_threshold`) Gaussians whose 3d │
    │     scale is below this value.                                                                 │
    │     These Gaussians are too small to capture the detail in the region they cover, so we        │
    │     duplicate them to                                                                          │
    │     allow them to specialize. (default: 0.01)                                                  │
    │ --opt.insertion-scale-2d-threshold FLOAT                                                       │
    │     Split high-error (determined by :obj:`insertion_grad_2d_threshold`) Gaussians whose        │
    │     maximum projected                                                                          │
    │     size exceeds this value. These Gaussians are too large to capture the detail in the region │
    │     they cover,                                                                                │
    │     so we split them to allow them to specialize.                                              │
    │                                                                                                │
    │                                                                                                │
    │     .. note:: This parameter is only used if set                                               │
    │     :obj:`use_screen_space_scales_for_refinement_until` is greater than 0. (default: 0.05)     │
    │ --opt.opacity-updates-use-revised-formulation,                                                 │
    │ --opt.no-opacity-updates-use-revised-formulation                                               │
    │     When splitting Gaussians, whether to update the opacities of the new Gaussians using the   │
    │     revised formulation from                                                                   │
    │     `*"Revising Densification in Gaussian Splatting"* <https://arxiv.org/abs/2404.06109>`_.    │
    │     This removes a bias which weighs newly split Gaussians contribution to the image more      │
    │     heavily than                                                                               │
    │     older Gaussians. (default: False)                                                          │
    │ --opt.insertion-split-factor INT                                                               │
    │     When splitting Gaussians during insertion, this value specifies the total number of new    │
    │     Gaussians that will                                                                        │
    │     replace each selected source Gaussian. The original is removed and replaced by             │
    │     :obj:`insertion_split_factor` new                                                          │
    │     Gaussians. *e.g.* if this value is 2, each split Gaussian is replaced by 2 new smaller     │
    │     Gaussians                                                                                  │
    │     (the original is removed). This value must be >= 2. (default: 2)                           │
    │ --opt.insertion-duplication-factor INT                                                         │
    │     When duplicating Gaussians during insertion, this value specifies the total number of      │
    │     copies (including                                                                          │
    │     the original) that will result for each selected source Gaussian. The original is kept,    │
    │     and                                                                                        │
    │     ``insertion_duplication_factor - 1`` new identical copies are added. *e.g.* if this value  │
    │     is 3,                                                                                      │
    │     each duplicated Gaussian becomes 3 copies of itself (the original plus 2 new). This value  │
    │     must be >= 2. (default: 2)                                                                 │
    │ --opt.reset-opacities-every-n-refinements INT                                                  │
    │     If set to a positive value, then clamp all opacities to be at most twice the value of      │
    │     :obj:`deletion_opacity_threshold` every time :func:`GaussianSplatOptimizer.refine` is      │
    │     called                                                                                     │
    │     :obj:`reset_opacities_every_n_refinements` times. This prevents Gaussians from becoming    │
    │     completely occluded by                                                                     │
    │     denser Gaussians and thus unable to be optimized. (default: 30)                            │
    │ --opt.use-scales-for-deletion-after-n-refinements INT                                          │
    │     If set to a positive value, then after ``use_scales_for_deletion_after_n_refinements``     │
    │     calls to                                                                                   │
    │     :func:`GaussianSplatOptimizer.refine`, use the 3D scales of the Gaussians to determine     │
    │     whether to delete them.                                                                    │
    │     This will delete Gaussians that have grown                                                 │
    │     too large in 3D space and are not contributing to the optimization.                        │
    │                                                                                                │
    │                                                                                                │
    │     By default, this value matches :obj:`reset_opacities_every_n_refinements` so that both     │
    │     behaviors are enabled at the                                                               │
    │     same time. (default: 30)                                                                   │
    │ --opt.use-screen-space-scales-for-refinement-until INT                                         │
    │     If set to a positive value, then use threshold the maximum projected size of Gaussians     │
    │     between refinement steps                                                                   │
    │     to decide whether to split or delete Gaussians that are too large. This behavior is        │
    │     enabled until                                                                              │
    │     :func:`GaussianSplatOptimizer.refine` has been called                                      │
    │     ``use_screen_space_scales_for_refinement_until`` times.                                    │
    │     After that, only 3D scales are used for refinement. (default: 0)                           │
    │ --opt.spatial-scale-mode                                                                       │
    │ {ABSOLUTE_UNITS,MEDIAN_CAMERA_DEPTH,MAX_CAMERA_DEPTH,MAX_CAMERA_TO_CENTROID,SCENE_DIAGONAL_PER │
    │ CENTILE}                                                                                       │
    │     How to interpret 3D optimization scale thresholds and learning rates (*i.e.*               │
    │     :obj:`insertion_scale_3d_threshold`,                                                       │
    │     :obj:`deletion_scale_3d_threshold`, and :obj:`means_lr`). These are scaled by a spatial    │
    │     scale computed from                                                                        │
    │     the scene, so they are relative to the size of the scene being optimized.                  │
    │                                                                                                │
    │                                                                                                │
    │     See :class:`SpatialScaleMode` for details on the available modes. (default:                │
    │     MEDIAN_CAMERA_DEPTH)                                                                       │
    │ --opt.spatial-scale-multiplier FLOAT                                                           │
    │     Multiplier to apply to the spatial scale computed from the scene to get a slightly larger  │
    │     scale. (default: 1.1)                                                                      │
    │ --opt.means-lr FLOAT                                                                           │
    │     Learning rate for the means of the Gaussians. This is also scaled by the spatial scale     │
    │     computed from the scene.                                                                   │
    │                                                                                                │
    │                                                                                                │
    │     See :obj:`spatial_scale_mode` for details on how the spatial scale is computed. (default:  │
    │     0.00016)                                                                                   │
    │ --opt.log-scales-lr FLOAT                                                                      │
    │     Learning rate for the log scales of the Gaussians. (default: 0.005)                        │
    │ --opt.quats-lr FLOAT                                                                           │
    │     Learning rate for the quaternions of the Gaussians. (default: 0.001)                       │
    │ --opt.logit-opacities-lr FLOAT                                                                 │
    │     Learning rate for the logit opacities of the Gaussians. (default: 0.05)                    │
    │ --opt.sh0-lr FLOAT                                                                             │
    │     Learning rate for the diffuse spherical harmonics (order 0). (default: 0.0025)             │
    │ --opt.shN-lr FLOAT                                                                             │
    │     Learning rate for the specular spherical harmonics (order > 0). (default: 0.000125)        │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ io options ───────────────────────────────────────────────────────────────────────────────────╮
    │ Configure saving and logging metrics, images, and checkpoints.                                 │
    │ ────────────────────────────────────────────────────────────────────────────────────────────── │
    │ --io.save-images, --io.no-save-images                                                          │
    │     Whether to save images to disk. If ``False``, images will not be saved to disk.            │
    │                                                                                                │
    │                                                                                                │
    │     Default is ``False``. (default: False)                                                     │
    │ --io.save-checkpoints, --io.no-save-checkpoints                                                │
    │     Whether to save checkpoints to disk. If ``False``, checkpoints will not be saved to disk.  │
    │                                                                                                │
    │                                                                                                │
    │     Default is ``True``. (default: True)                                                       │
    │ --io.save-plys, --io.no-save-plys                                                              │
    │     Whether to save PLY files to disk. If ``False``, PLY files will not be saved to disk.      │
    │                                                                                                │
    │                                                                                                │
    │     Default is ``True``. (default: True)                                                       │
    │ --io.save-metrics, --io.no-save-metrics                                                        │
    │     Whether to save metrics to a CSV file. If ``False``, metrics will not be saved to a CSV    │
    │     file.                                                                                      │
    │                                                                                                │
    │                                                                                                │
    │     Default is ``True``. (default: True)                                                       │
    │ --io.metrics-file-buffer-size INT                                                              │
    │     How much buffering (in bytes) to use for metrics file logging. Larger values can improve   │
    │     performance when logging many metrics.                                                     │
    │                                                                                                │
    │                                                                                                │
    │     Default is 8 MiB. (default: 8388608)                                                       │
    │ --io.use-tensorboard, --io.no-use-tensorboard                                                  │
    │     Whether to use TensorBoard for logging metrics and images. If ``True``, metrics and images │
    │     will be logged to TensorBoard.                                                             │
    │                                                                                                │
    │                                                                                                │
    │     Default is ``False``. (default: False)                                                     │
    │ --io.save-images-to-tensorboard, --io.no-save-images-to-tensorboard                            │
    │     Whether to also save images to TensorBoard if :obj:`use_tensorboard` is ``True``. If       │
    │     ``True``, images will be saved to TensorBoard.                                             │
    │                                                                                                │
    │                                                                                                │
    │     Default is ``False``. (default: False)                                                     │
    │ --io.log-path {None}|PATH                                                                      │
    │     Path to save logs, checkpoints, and other output to. Defaults to `frgs_logs` in the        │
    │     current working directory. (default: frgs_logs)                                            │
    │ --io.log-every INT                                                                             │
    │     How frequently to log metrics during reconstruction. (default: 10)                         │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

