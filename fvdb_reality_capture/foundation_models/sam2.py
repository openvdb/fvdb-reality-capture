# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import sam2.utils.amg as amg
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .config import get_weights_path_for_model


class SAM2Model:
    """
    Wrapper for SAM2 supporting flat and multi-scale mask generation.

    SAM2 is a foundation model solving promptable visual segmentation in images and videos.

    The original SAM2 paper can be found here:
        https://arxiv.org/abs/2408.00714

    - **output_mode="flat"** (default): One list of masks per image via
      ``predict_masks(image)``.
    - **output_mode="multi_scale"**: One image/crop and one set of points in,
      four lists out (default / small / medium / large). Callers implement
      crops and merging (e.g. LangSplatV2-style preprocessing).
    """

    def __init__(
        self,
        checkpoint: Literal["large", "small", "tiny", "base_plus"] = "large",
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.8,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        output_mode: Literal["flat", "multi_scale"] = "flat",
        device: torch.device | str = "cuda",
    ):
        """
        Initialize the SAM2 model.

        Args:
            checkpoint: Checkpoint size (large, small, tiny, base_plus).
            points_per_side: Grid density for point prompts.
            points_per_batch: Points per batch; only used when output_mode="multi_scale".
            pred_iou_thresh: Minimum predicted IoU for keeping a mask.
            stability_score_thresh: Minimum stability score for keeping a mask.
            stability_score_offset: Offset for stability score; multi_scale only.
            mask_threshold: Binarization threshold; multi_scale only.
            box_nms_thresh: Box NMS IoU threshold within the image; multi_scale only.
            min_mask_region_area: Min mask region area for predictor; multi_scale only.
            output_mode: "flat" or "multi_scale".
            device: Device to run on.
        """
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._output_mode = output_mode
        self._device = torch.device(device)
        self._checkpoint = checkpoint
        self._points_per_side = points_per_side
        self._points_per_batch = points_per_batch
        self._pred_iou_thresh = pred_iou_thresh
        self._stability_score_thresh = stability_score_thresh
        self._stability_score_offset = stability_score_offset
        self._mask_threshold = mask_threshold
        self._box_nms_thresh = box_nms_thresh
        self._min_mask_region_area = min_mask_region_area

        self._logger.info(f"Loading SAM2 model ({checkpoint}, output_mode={output_mode})")

        # Download the checkpoint
        ckpt_name = f"sam2.1_hiera_{checkpoint}.pt"
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ckpt_name}"
        path_to_weights = get_weights_path_for_model(ckpt_name, url, model_name="SAM2")
        config_name_map = {"large": "l", "small": "s", "tiny": "t", "base_plus": "b+"}
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{config_name_map[checkpoint]}"

        # Build the model
        self._sam2_base = build_sam2(
            model_cfg,
            ckpt_path=path_to_weights,
            device=str(self._device),
            apply_postprocessing=False,
        )

        if output_mode == "flat":
            self._mask_generator = SAM2AutomaticMaskGenerator(
                self._sam2_base,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
            self._predictor = None
        else:
            self._mask_generator = None
            self._predictor = SAM2ImagePredictor(
                self._sam2_base,
                max_hole_area=min_mask_region_area,
                max_sprinkle_area=min_mask_region_area,
            )

        self._logger.info("SAM2 model loaded successfully.")

    def predict_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict masks for an image (flat list).

        Use when output_mode="flat". Image shape [H, W, C], uint8.

        Returns:
            List of mask records (segmentation, bbox, area, predicted_iou,
            point_coords, stability_score, crop_box).
        """
        if self._output_mode != "flat":
            raise RuntimeError(
                "predict_masks() is for output_mode='flat'. "
                "Use predict_masks_multi_scale() for output_mode='multi_scale'."
            )
        return self._mask_generator.generate(image)

    @torch.no_grad()
    def predict_masks_multi_scale(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        crop_box: Optional[List[int]] = None,
        orig_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Predict masks for one image (or crop) with scale split by multimask index.

        Runs the predictor on the given image and point prompts, then splits
        the 3 multimask outputs per point into default / small / medium / large.
        No crop strategy is applied here; callers may pass a cropped image and
        crop_box/orig_size to get masks in full-image coordinates.

        Args:
            image: Image or crop, HWC uint8 (e.g. RGB).
            point_coords: Point prompts in image space, shape [N, 2] (x, y).
                If None, a single full-image grid is built using points_per_side.
            crop_box: [x0, y0, x1, y1] of this image in original frame. If None,
                image is treated as full image and crop_box = [0, 0, W, H].
            orig_size: (height, width) of the full image. If None, set to
                image.shape[:2]. Required when crop_box is provided.

        Returns:
            (masks_default, masks_s, masks_m, masks_l) — lists of mask records.
        """
        if self._output_mode != "multi_scale":
            raise RuntimeError(
                "predict_masks_multi_scale() is for output_mode='multi_scale'. "
                "Use predict_masks() for output_mode='flat'."
            )
        im_h, im_w = image.shape[:2]
        if orig_size is None:
            orig_size = (im_h, im_w)
        if crop_box is None:
            crop_box = [0, 0, im_w, im_h]

        if point_coords is None:
            grid = amg.build_all_layer_point_grids(self._points_per_side, 0, 1)[0]
            points_scale = np.array([im_w, im_h], dtype=np.float64)
            point_coords = grid * points_scale

        data_all, data_s, data_m, data_l = self._process_image(image, point_coords, crop_box, orig_size)

        for data in (data_all, data_s, data_m, data_l):
            data.to_numpy()

        return (
            self._to_ann_list(data_all),
            self._to_ann_list(data_s),
            self._to_ann_list(data_m),
            self._to_ann_list(data_l),
        )

    def _process_image(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        crop_box: List[int],
        orig_size: Tuple[int, int],
    ) -> Tuple[Any, Any, Any, Any]:
        """Run the predictor on one image/crop and split by scale index.

        Args:
            image: Image or crop, HWC uint8.
            point_coords: Points in image space [N, 2].
            crop_box: [x0, y0, x1, y1] in full-image frame.
            orig_size: (height, width) of full image.

        Returns:
            (data_all, data_s, data_m, data_l) MaskData.
        """
        im_size = image.shape[:2]
        self._predictor.set_image(image)

        data_all = amg.MaskData()
        data_s = amg.MaskData()
        data_m = amg.MaskData()
        data_l = amg.MaskData()

        for (points,) in amg.batch_iterator(self._points_per_batch, point_coords):
            bd_all, bd_s, bd_m, bd_l = self._process_batch(points, im_size, crop_box, orig_size)
            data_all.cat(bd_all)
            data_s.cat(bd_s)
            data_m.cat(bd_m)
            data_l.cat(bd_l)

        self._predictor.reset_predictor()

        from torchvision.ops.boxes import batched_nms

        for data in (data_all, data_s, data_m, data_l):
            if len(data["rles"]) == 0:
                continue
            keep = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self._box_nms_thresh,
            )
            data.filter(keep)
            data["boxes"] = amg.uncrop_boxes_xyxy(data["boxes"], crop_box)
            data["points"] = amg.uncrop_points(data["points"], crop_box)
            data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data_all, data_s, data_m, data_l

    def _to_ann_list(self, data) -> List[Dict[str, Any]]:
        """Convert MaskData (rles, boxes, etc.) to a list of mask-record dicts.

        Returns:
            List of dicts with segmentation, area, bbox, predicted_iou,
            point_coords, stability_score, crop_box.
        """
        if len(data["rles"]) == 0:
            return []
        segmentations = [amg.rle_to_mask(rle) for rle in data["rles"]]
        anns: List[Dict[str, Any]] = []
        for idx in range(len(segmentations)):
            anns.append(
                {
                    "segmentation": segmentations[idx],
                    "area": amg.area_from_rle(data["rles"][idx]),
                    "bbox": amg.box_xyxy_to_xywh(data["boxes"][idx]).tolist(),
                    "predicted_iou": data["iou_preds"][idx].item(),
                    "point_coords": [data["points"][idx].tolist()],
                    "stability_score": data["stability_score"][idx].item(),
                    "crop_box": amg.box_xyxy_to_xywh(data["crop_boxes"][idx]).tolist(),
                }
            )
        return anns

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, int],
        crop_box: List[int],
        orig_size: Tuple[int, int],
    ) -> Tuple[Any, Any, Any, Any]:
        """Run SAM2 on a batch of point prompts and split the 3 multimask outputs by index.

        Args:
            points: Point coordinates in image/crop space, shape [N, 2].
            im_size: (height, width) of the image/crop.
            crop_box: [x0, y0, x1, y1] in full-image frame.
            orig_size: (height, width) of the full image.

        Returns:
            (data_all, data_s, data_m, data_l) MaskData. Index 0 = small, 1 = medium, 2 = large.
        """
        orig_h, orig_w = orig_size

        pts = torch.as_tensor(points, dtype=torch.float32, device=self._predictor.device)
        in_points = self._predictor._transforms.transform_coords(pts, normalize=True, orig_hw=im_size)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=pts.device)

        masks, iou_preds, _ = self._predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        scale_datas = []
        for scale_idx in range(masks.shape[1]):
            sd = amg.MaskData(
                masks=masks[:, scale_idx, :, :],
                iou_preds=iou_preds[:, scale_idx],
                points=pts.clone(),
            )
            scale_datas.append(sd)

        data_all = amg.MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=pts.repeat_interleave(masks.shape[1], dim=0),
        )
        del masks

        for data in [data_all, *scale_datas]:
            if self._pred_iou_thresh > 0.0:
                keep = data["iou_preds"] > self._pred_iou_thresh
                data.filter(keep)
            data["stability_score"] = amg.calculate_stability_score(
                data["masks"],
                self._mask_threshold,
                self._stability_score_offset,
            )
            if self._stability_score_thresh > 0.0:
                keep = data["stability_score"] >= self._stability_score_thresh
                data.filter(keep)
            data["masks"] = data["masks"] > self._mask_threshold
            data["boxes"] = amg.batched_mask_to_box(data["masks"])
            keep = ~amg.is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
            if not torch.all(keep):
                data.filter(keep)
            data["masks"] = amg.uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
            data["rles"] = amg.mask_to_rle_pytorch(data["masks"])
            del data["masks"]

        return data_all, scale_datas[0], scale_datas[1], scale_datas[2]
