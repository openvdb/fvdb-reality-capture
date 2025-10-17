# Welcome to fVDB-Reality-Capture!

fVDB-Reality-Capture is a library for building reality capture applications that reconstruct and process 3D capture
data from sensors built with [fVDB](https://openvdb.github.io/fvdb).
fVDB-Reality-Capture aims to be production ready, with a focus on robustness, usability, and extensibility.
It is designed to be easily integrated into existing pipelines and workflows, and to support a wide range of use cases and applications.

  <video autoplay loop controls muted width="100%">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/Large_World_480p.mp4" type="video/mp4" />
  </video>




## What is Reality Capture?
Reality capture is the process of creating digital 3D representations of real-world objects and environments using
various sensing technologies, such as cameras, LiDAR, and depth sensors. The goal of reality capture is to
accurately and efficiently reconstruct the geometry, appearance, and other properties of the physical world into usable 3D models and scenes.
Modern reality capture pipelines make heavy use of [radiance fields](https://radiancefields.com/)
and fVDB-Reality-Capture provides best-in-class tools to reconstruct radiance fields using
[3D Gaussian Splatting](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/3d-vision/3d-gaussian-splatting/).
A common reality capture pipeline typically resembles the figure below:

  <img src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/meshing_med.gif"
       alt="Reality Capture Pipeline"
       style="display: block; margin-left: auto; margin-right: auto; width: 100%; margin-top: -4em; margin-bottom: -2em" />




## Installation
To get started, simply run

```bash
pip install fvdb-reality-capture
```




# Library Overview
Analogous to how [torchvision](https://docs.pytorch.org/vision/stable/index.html) builds a computer-vision specialized
toolbox on top of [PyTorch](https://pytorch.org/), fVDB-Reality-Capture builds a reality-capture specialized toolbox on top of fVDB.

With fVDB-Reality-Capture, you can easily load and manipulate 3D capture data, reconstruct radiance fields with 3D Gaussian splatting,
and extract high quality meshes and point clouds. All these can be visualized in a browser or notebook, and exported to common formats
like PLY and USDZ.

fVDB-Reality-Capture is built on top of the [fVDB](https://openvdb.github.io/fvdb) library, which provides efficient GPU data structures and algorithms for
working with sparse volumetric data. By leveraging the power of fVDB, fVDB-Reality-Capture can scale reconstruction
to very large or dense inputs, while maintaining high performance and low memory usage.

fVDB-Reality-Capture aims to be production ready, with a focus on robustness, usability, and extensibility. It is designed to be easily integrated into existing
pipelines and workflows, and to support a wide range of use cases and applications. To this end, both fVDB and fVDB-Reality-Capture
have a minimal set of dependencies, and are open source under the Apache 2.0 license. We welcome contributions and feedback from the community.

Note: fVDB-Reality-Capture recently moved from early-access to Beta. The API and documentation are still subject to improvements as we move towards a 1.0 release.

