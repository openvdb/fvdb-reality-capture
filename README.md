# Welcome to fVDB-Reality-Capture!

fVDB-Reality-Capture is a reality-capture toolbox built on top of [fVDB](https://fvdb.ai). It
provides high-level abstractions and APIs for common reality capture tasks, such as loading sensor data, reconstructing
radiance fields, extracting meshes and point clouds, visualization, and exporting results across standard formats such
as PLY and USDZ.

By leveraging the power of fVDB, fVDB-Reality-Capture can scale reconstruction to very large or dense
inputs, while maintaining high performance and low memory usage. *fVDB has 50% better throughput than gsplat in end-to-end training benchmarks and 30% lower runtime, while producing higher quality results and working out-of-the box on a wide range of inputs*.
The videos below show large-scale reconstructions of complex scenes using fVDB-Reality-Capture.


  <video autoplay loop controls muted width="100%">
     <source src="https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/Large_World_480p.mp4" type="video/mp4" />
  </video>

----

**For more information about what fVDB-Reality-Capture can do, tutorials and documentation, please see the
[fVDB-Reality-Capture documentation](https://fvdb.ai/reality-capture/).**



## Installation
To get started, simply run

```bash
pip install fvdb-reality-capture fvdb-core==0.3.0+pt28.cu128 --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

To install fvdb-reality-capture as well as the appropriate version of fvdb-core and torch.

### Installation from source

Clone the [fvdb-reality-capture repository](https://github.com/openvdb/fvdb-reality-capture). Then:

```bash
cd fvdb-reality-capture
pip install -e . # for non-editable, drop the -e
```

# Library Overview
fVDB-Reality-Capture is a reality-capture-specialized toolbox on top of fVDB analogous to how [torchvision](https://docs.pytorch.org/vision/stable/index.html) is a computer-vision-specialized toolbox on top of [PyTorch](https://pytorch.org/).


fVDB-Reality-Capture is built on top of [fVDB](https://openvdb.github.io/fvdb), which provides efficient GPU data
structures and algorithms for working with sparse volumetric data. By leveraging the power of fVDB, fVDB-Reality-Capture
can scale reconstruction to very large or dense inputs, while maintaining high performance and low memory usage.

fVDB-Reality-Capture aims to be production ready, with a focus on robustness, usability, and extensibility. It is
designed to be easily integrated into existing pipelines and workflows, and to support a wide range of use cases and applications. To this end, both fVDB and fVDB-Reality-Capture have a minimal set of dependencies, and are open source
under the Apache 2.0 license. We welcome contributions and feedback from the community.
