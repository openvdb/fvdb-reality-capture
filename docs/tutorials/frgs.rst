Reconstruction on the CLI with ``frgs`` 🐸
=============================================================

fVDB-Reality-Capture provides a command-line interface (CLI) tool called ``frgs`` for tools related to Gaussian splatting.
``frgs`` is pronounced "frogs" 🐸, and is short for  **F**\ vdb-\ **R**\ eality-capture **G**\ aussian **S**\ platting.

This tool allows users to perform Gaussian splat related tasks such as reconstruction, format conversions,
mesh and point cloud extraction, and visualization directly from the command line, without needing to write any code.

We'll give a brief tutorial on how to use ``frgs`` here, and provide a full reference of all available commands and options
in the CLI Command Reference section below.


Download Example Data
---------------------------------

Let's start by downloading some example data to work with. We'll use the ``frgs download`` command to download a sample dataset:
We'll use a simple dataset from the `Mip-NeRF 360 <https://arxiv.org/abs/2111.12077>`_ collection. These are small and
so relatively quick to download and reconstruct. While we usually show large high quality outdoor captures in our results,
this demo showcases a low quality indoor capture using a handheld phone, which is more challenging but also a common use case.
You can of course try out other datasets as well!

.. code-block:: bash

   frgs download --dataset "mipnerf360"

This will output a progress bar as the dataset is downloaded and extracted to a directory called `data/360_v2/` in your current working directory.

.. code-block:: bash

   INFO - Downloading dataset mipnerf360 from https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/360_v2.zip to /home/fwilliams/projects/openvdb/fvdb-reality-capture/data/360_v2/360_v2.zip
   Downloading dataset mipnerf360:   100%|████████████████████████████| 728M/12.5G [10:11<00:00, 18.8MB/s]
   INFO - Extracting dataset mipnerf360 to /home/fwilliams/projects/openvdb/fvdb-reality-capture/data/360_v2/
   INFO - Dataset mipnerf360 downloaded and extracted to /home/fwilliams/projects/openvdb/fvdb-reality-capture/data/360_v2/


Visualize our Data
---------------------------------

Next let's look at one of our datasets to see what it contains. We can use the ``frgs show-data`` command to
show the camera poses and input sparse points for a dataset.

.. code-block:: bash

    frgs show-data ./data/360_v2/room

which should pop open a browser window with a 3D viewer showing the camera poses and sparse points.
You can navigate the scene using mouse controls (left-click to rotate, right-click to pan, scroll to zoom).

.. raw:: html

  <video autoplay loop controls muted width="100%">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/frgs_points.mov" type="video/mp4" />
  </video>


Reconstruct a Gaussian Splat Radiance Field
---------------------------------------------

Our capture looks reasonable a few sweeps around parts of the room, but there are some gaps and missing areas.
Let's reconstruct a Gaussian splat radiance field from this data using the ``frgs reconstruct`` command:

.. code-block:: bash

   frgs reconstruct ./data/360_v2/room -o room.ply

This will start the reconstruction process, which may take some time depending on the size of the dataset and your hardware.
You should see output similar to the following, with progress updates as the reconstruction proceeds:

.. code-block:: bash

    INFO - Loading dataset from data/360_v2/room
    INFO - Loading visible points per image from cache...
    INFO - Normalizing SfmScene with normalization type: pca
    INFO - Filtering points based on percentiles: min=[0. 0. 0.], max=[100. 100. 100.]
    INFO - No points will be filtered out, returning the input scene unchanged.
    INFO - Rescaling images using downsample factor 4, sampling mode 3, and quality 95.
    INFO - Attempting to load downsampled images from cache.
    INFO - Created unique log directory with name run_2025-10-17-15-31-18 after 0 attempts.
    INFO - Created training and validation datasets with 311 training images and 0 validation images.
    INFO - Model initialized with 112,627 Gaussians
    Reconstructing:   0%|                                                               | 0/62200 [00:00<?, ?imgs/s]
    INFO - Starting to optimize camera poses at step 0 (epoch 0)
    loss=0.017| sh degree=3| num gaussians=1,163,393:  20%|█▉        | 12439/62200 [01:35<08:10, 101.53imgs/s]
    INFO - Saving checkpoint at global step 12440.
    loss=0.025| sh degree=3| num gaussians=1,175,393:  23%|██▍        | 14066/62200 [01:55<08:45, 91.63imgs/s]
    ...

When the process completes, you should have a file called `room.ply` in your current working directory.
You'll also also have a log directory called `frgs_logs/run_YYYY-MM-DD-HH-MM-SS/` containing checkpoints and training logs
which you can analyze later.

Visualize the Reconstructed Radiance Field
---------------------------------------------

You can visualize the reconstructed Gaussian splat radiance field using the ``frgs show`` command:

.. code-block:: bash

   frgs show room.ply

.. raw:: html

  <video autoplay loop controls muted width="100%">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/frgs_recon.mov" type="video/mp4" />
  </video>

which will pop up a viewer in your browser where you can explore the reconstructed scene.


Extract a Mesh from the Radiance Field
---------------------------------------------

Now let's see how to extract a mesh from the reconstructed Gaussian splat radiance field.
We'll use the ``frgs mesh-dlnr`` command to extract a high quality mesh from the radiance field.

This algorithm requires you pick a truncation margin which defines the width of a narrow band
around the surface where the density field is evaluated for meshing. This should typically be set to around
6 times your target mesh resolution (e.g., if you want a mesh with 1cm resolution, set the margin to around 6cm).

Here we'll extract a mesh at around 1.5cm resolution by setting the truncation margin to 10cm:

.. code-block:: bash

   frgs mesh-dlnr room.ply -o room_mesh.ply 0.10


This produces a mesh which looks something like this:

.. raw:: html

  <img src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/room_mesh.png"
       alt="Reality Capture Pipeline"
       style="display: block; margin-left: auto; margin-right: auto; width: 100%;" />



