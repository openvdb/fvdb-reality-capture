**Evaluation**

To evaluate GARfVDB, we're going to use the [GARField evaluations from their paper](https://drive.google.com/drive/folders/1LDvbFTQuaQxru5ELsfCjX7sTkg1WotX0).  To do this, we'll need to train a GARfVDB model on the same data as the GarFIELD model and this requires jumping throuhg some hoops compared to starting from a set of captured images in order to make sure we're training and evaluating on the same data.

## 1. Convert the nerfstudio scene to colmap format

First the nerfstudio camera poses are converted to colmap format and space:

```bash
python nerfstudio_to_colmap.py --input_dir /path/to/nerfstudio/input_scene --output_dir /path/to/colmap/sparse/0
```

Next we need to generate the points3D.bin with colmap to seed the gaussian splatting model.

First we need to extract features from the images.
NOTE:  You may want to set COLMAP to run headless with:
```bash
export QT_QPA_PLATFORM=offscreen
```
```bash
colmap feature_extractor \
  --database_path /path/to/colmap//database.db \
  --image_path /path/to/colmap/images
```

```bash
colmap exhaustive_matcher \
  --database_path /path/to/colmap/database.db
```

```bash
colmap point_triangulator \
  --database_path /path/to/colmap/database.db \
  --image_path /path/to/colmap/images \
  --input_path /path/to/colmap/sparse/0 \
  --output_path /path/to/colmap/sparse/0 (or another output path)
```

## 2. Train a 3dgs model on the colmap scene

```bash
python train.py --dataset-path /path/to/colmap/sparse/0 --image-downsample-factor 1
```

## 3. Generate the same masks used in GARField evaluation to re-use in GARfVDB

At the moment, I've been using GARField to generate masks since they use some specific parameters
for their evaluation that are different from default.

NOTE:  For this step, I've been manually editing the GARField code to generate masks for all the
images in the scene instead of just the ones in the training set as is the default.  To do, we
should change the make_segmentation_dataset.py script to be able to generate masks with the same
parameters as GARField.

```bash
ns-train garfield \
  --pipeline.datamanager.img-group-model.model-type sam_fb \
  --pipeline.datamanager.img-group-model.sam-model-type vit_h \
  --pipeline.datamanager.img-group-model.sam-model-ckpt /path/to/models/sam_vit_h_4b8939.pth \
  --pipeline.datamanager.img-group-model.sam-kwargs points_per_side 32  pred_iou_thresh 0.90 stability_score_thresh 0.90  \
  --data /path/to/LERF/Datasets/ramen
```

Now we need to convert the masks to the format we use in GARfVDB.

This script will perform this conversion as well as re-match the images to the colmap scene cameras
and needs to use the 3dgs model checkpoint to determine mask scales based on the positiions in the
3dgs model.

```bash
python make_segmentation_dataset.py \
  --garfield-sam-data-path /path/to/garfield/outputs/ramen/sam_data.hdf5  \
  --checkpoint-path /path/to/3dgs/checkpoints/ckpt_29999.pt  \
  --colmap-path /path/to/colmap/scene/ \
  --output-filepath ./segmentation_dataset.pt
```


## 4. Train a GARfVDB model on the same data as the GarFIELD model

```bash
python train_segmentation.py  \
  --checkpoint-path /path/to/3dgs/checkpoints/ckpt_39999.pt  \
  --segmentation_dataset_path  ./segmentation_dataset.pt  \
  --config.sample-pixels-per-image 1024 \
  --config.batch-size 1 \
  --config.model.depth-samples 24 \
  --config.accumulate-grad-steps 8
```

## 5. Evaluate the GARfVDB model

```bash
python eval_completeness.py \
  --garfvdb-checkpoint-path /path/to/garfvdb/checkpoints/segmentation_checkpoint_9999.pt \
  --gsplat-checkpoint-path /path/to/3dgs/checkpoints/ckpt_39999.pt  \
  --segmentation-dataset-path ./segmentation_dataset.pt  \
  --eval-data-path /path/to/garfield_eval/eval_data/  \
  --scene-name ramen \
  --results-path results/garfvdb_completeness \
  --garfield-output-path /path/to/garfield/outputs/ramen/garfield/2025-07-23_000716/ \
  --colmap-dataset-path /path/to/colmap/scene/
```
