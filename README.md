# 3D Gaussian Splatting with ƒVDB

1. Prereqs:
   fvdb conda environment with ƒVDB installed. See the
   [ƒVDB README](https://github.com/NVIDIA-Omniverse/openvdb/blob/feature/fvdb/fvdb/README.md).

2. Setup the environment:

    ```bash
    conda activate fvdb
    ```

3. Download the example data

    ```bash
    ./download_example-data.py
    ```

4. Run the `train_colmap.py` example
    ```bash
    python train_colmap.py --data-path data/360_v2/[scene_name]
    ```

5. View the results in a browser at `http://localhost:8080`
