# Welcome to fVDB-Reality-Capture!

fVDB-Reality-Capture is a reality-capture toolbox built on top of [fVDB](https://fvdb.ai). It
provides high-level abstractions and APIs for common reality capture tasks, such as loading sensor data, reconstructing
radiance fields, extracting meshes and point clouds, visualization, and exporting results across standard formats such
as PLY and USDZ.

By leveraging the power of fVDB, fVDB-Reality-Capture can scale reconstruction to very large or dense
inputs, while maintaining high performance and low memory usage. *fVDB has 50% better throughput than gsplat in end-to-end training benchmarks and 30% lower runtime, while producing higher quality results and working out-of-the-box on a wide range of inputs*.
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

> **Note:**
> For more complete instructions for building `fvdb-core` from source, including setting up a build environment and
> obtaining the necessary dependencies, see the fVDB [README](https://github.com/openvdb/fvdb-core/blob/main/README.md).


Clone the [fvdb-core repository](https://github.com/openvdb/fvdb-core) and the [fvdb-reality-capture repository](https://github.com/openvdb/fvdb-reality-capture).

```bash
git clone git@github.com:openvdb/fvdb-core.git
git clone git@github.com:openvdb/fvdb-reality-capture.git
```

Next, build and install the fVDB library

```bash
pushd fvdb-core
./build.sh install verbose editor_force
popd
```

Finally, install fVDB-Reality-Capture

```bash
pushd fvdb-reality-capture
pip install .
popd
```


# About fVDB-Reality-Capture

fVDB and fVDB-Reality-Capture were first developed by the
[NVIDIA High-Fidelity Physics Research Group](https://research.nvidia.com/labs/prl/)
within the [NVIDIA Spatial Intelligence Lab](https://research.nvidia.com/labs/sil/), and continues to be
developed with the OpenVDB community to suit the growing needs for a robust framework for
spatial intelligence research and applications.

fVDB-Reality-Capture is built on top of [fVDB](https://github.com/openvdb/fvdb-core), which provides efficient GPU data
structures and algorithms for working with sparse volumetric data. By leveraging the power of fVDB, fVDB-Reality-Capture
can scale reconstruction to very large or dense inputs, while maintaining high performance and low memory usage.

fVDB and fVDB-Reality-Capture are open source under the Apache 2.0 license. We welcome contributions and feedback
from the community.

For questions or feedback, please use the [GitHub Issues](https://github.com/openvdb/fvdb-reality-capture/issues) for this repository.
