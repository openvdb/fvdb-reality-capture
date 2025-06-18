## Setup Environment

```bash
conda env create -f ./garfvdb_environment.yml
conda activate fvdb_garfvdb
```

### Build and Install ƒVDB wheel

```bash
./build.sh wheel
pip install ./dist/[fvdb wheel]
```

## GARƒVDB on GARField's Dozer, Nerf Gun, and Waldo

### Train Gaussian Splat Scene on Scene

1. Download garfield example data
    ```bash
    python ../../panoptic_segmentation/garfield/download_example_data.py
    ```
1. Train 3d gaussian splatting on the dozer, nerf gun, and waldo scene
    ```bash
    python ../training/train_colmap.py --data-path data/dozer_nerfgun_waldo --up-axis="-z"
    ```

### Make Segmentation Dataset

```bash
python make_segmentation_dataset.py --checkpoint-path ../training/results/[your 3dgs run]/checkpoints/ckpt_29999.pt  --colmap-path data/dozer_nerfgun_waldo/ --output-path segmentation_dataset.pt
```

### Train GARƒVDB

1. Run the following command to train GARƒVDB
```bash
 python train_segmentation.py --checkpoint-path ../training/results/[your 3dgs run]/checkpoints/ckpt_29999.pt --segmentation_dataset_path  ./segmentation_dataset.pt
 ```

2. Launch Tensorboard to monitor the training progress
```bash
tensorboard --logdir ./logs
```

3. Go to http://localhost:6006/ to monitor the training progress
