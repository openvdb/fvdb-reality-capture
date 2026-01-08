Welcome to ƒVDB-Reality-Capture!
===================================================

.. raw:: html

  <video autoplay loop controls muted width="90%" style="display: block; margin: 0 auto;">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/rc_intro_480p.mp4" type="video/mp4" />
  </video>

|

ƒVDB-Reality-Capture is a reality-capture toolbox built on top of `ƒVDB <https://fvdb.ai>`_. It is maintained and developed by NVIDIA. ƒVDB-Reality-Capture provides high-level abstractions and APIs for common reality capture tasks, such as loading sensor data, reconstructing
radiance fields, extracting meshes and point clouds, visualization, and exporting results across standard formats such
as PLY and USDZ. By leveraging the power of ƒVDB, ƒVDB-Reality-Capture can scale reconstruction to very large or dense
inputs, while maintaining high performance and low memory usage. *ƒVDB has 50% better throughput than gsplat in end-to-end training benchmarks and 30% lower runtime, while producing higher quality results and working out-of-the box on a wide range of inputs*.
The videos below show large-scale reconstructions of complex scenes using ƒVDB-Reality-Capture.

.. raw:: html

   <p style="text-align: center; font-weight: bold; font-style: italic; text-decoration: underline; font-size: medium; text-decoration-skip-ink: none; margin-bottom: 0.5em;">
   100 million 3D Gaussians reconstructed from 400 high-resolution images, rendered in a browser</p>
  <video autoplay loop controls muted width="90%"  style="display: block; margin: 0 auto;">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/large_recon_480p.mp4" type="video/mp4" />
  </video>

|

.. raw:: html

   <p style="text-align: center; font-weight: bold; font-style: italic; text-decoration: underline; font-size: medium; text-decoration-skip-ink: none; margin-bottom: 0.5em;">
   Reconstructing a high quality Gaussian splat radiance field and mesh from 177 images</p>
  <video autoplay loop controls muted width="90%" style="display: block; margin: 0 auto;">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/airport_viz_480p.mp4" type="video/mp4" />
  </video>

|

The relationship between ƒVDB-Reality-Capture and ƒVDB is analogous to the relationship between
`torchvision <https://docs.pytorch.org/vision/stable/index.html>`_ and `PyTorch <https://pytorch.org/>`_, where
ƒVDB-Reality-Capture provides domain-specific functionality for reality capture applications, while ƒVDB provides the
underlying data structures and algorithms for efficient GPU computation.

ƒVDB-Reality-Capture aims to be production ready, with a focus on robustness, usability, and extensibility. It is
designed to be easily integrated into existing pipelines and workflows, and to support a wide range of use cases and
applications. To this end, both ƒVDB and ƒVDB-Reality-Capture have a minimal set of dependencies, and are open source
under the Apache 2.0 license. We welcome contributions and feedback from the community.

Features
-----------
- Efficient loading and manipulation of sensor data (images, depth maps, camera poses, etc.), even if they are larger
  than available memory.
- Composable transforms for common data augmentation and preprocessing operations, with built-in caching for improved
  performance.
- State-of-the-art radiance field reconstruction using 3D Gaussian Splatting, optimized for speed and quality.
- Extraction of high-quality meshes and point clouds from reconstructed radiance fields, with support for various output
  formats.
- Visualization tools for inspecting sensor data, radiance fields, and reconstruction results.
- Extensible design that allows users to easily add new functionality and customize existing components.
- Comprehensive documentation and tutorials to help users get started quickly.
- Automatic scaling to multiple GPUs in a single machine for large-scale reconstructions
  (source code only, packaging coming soon).

What is Reality Capture?
-----------------------------
Reality capture is the process of creating digital 3D representations of real-world objects and environments using
various sensing technologies, such as cameras, LiDAR, and depth sensors. The goal of reality capture is to accurately
and efficiently reconstruct the geometry, appearance, and other properties of the physical world into usable 3D models
and scenes.

Modern reality capture pipelines make heavy use of `radiance fields <https://radiancefields.com/>`_.
ƒVDB-Reality-Capture provides best-in-class tools to reconstruct radiance fields using
`3D Gaussian Splatting <https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/3d-vision/3d-gaussian-splatting/>`_.

A common reality capture pipeline typically resembles the figure below:

.. raw:: html

  <img src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/meshing_med.gif"
       alt="Reality Capture Pipeline"
       style="display: block; margin-left: auto; margin-right: auto; width: 100%;" />


.. toctree::
   :caption: Introduction
   :hidden:

   self
   installation
   GitHub Repository <https://github.com/openvdb/fvdb-reality-capture>

.. toctree::
   :maxdepth: 1
   :caption: Tutorials and Examples

   tutorials/sensor_data_loading_and_manipulation
   tutorials/radiance_field_and_mesh_reconstruction
   tutorials/frgs

.. toctree::
   :maxdepth: 1
   :caption: API References

   api/radiance_fields
   api/sfm_scene
   api/tools
   api/transforms
   api/frgs

.. toctree::
   :maxdepth: 1
   :caption: fVDB-Core Library
   :hidden:

   fvdb-core Docs <https://fvdb.ai/docs/>
   fvdb-core GitHub <https://github.com/openvdb/fvdb-core>

.. raw:: html

   <hr>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
