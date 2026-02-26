# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from .config import get_weights_path_for_model

_SAM1_INSTALL_MSG = (
    "SAM1 requires the segment-anything package. Install with:\n"
    "  conda install -c conda-forge segment-anything\n"
    "or \n"
    "  pip install 'git+https://github.com/facebookresearch/segment-anything.git'\n"
)

_SAM1_CHECKPOINTS: Dict[str, Tuple[str, str, str]] = {
    "vit_h": (
        "vit_h",
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
    "vit_l": (
        "vit_l",
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_b": (
        "vit_b",
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
}


class SAM1Model:
    """
    Wrapper for SAM1 (Segment Anything Model) supporting multi-scale mask generation.

    The original SAM paper can be found here:
        https://arxiv.org/abs/2304.02643

    - **output_mode="multi_scale"**: One image/crop and one set of points in,
      four lists out (default / small / medium / large). Callers implement
      crops and merging (e.g. LangSplatV2-style preprocessing).

    The multi-scale output splits SAM's 3 multimask outputs per point prompt
    (small / medium / large) into separate lists, matching the approach used by
    the original LangSplatV2's forked SAM mask generator.
    """

    def __init__(
        self,
        checkpoint: Literal["vit_h", "vit_l", "vit_b"] = "vit_h",
        points_per_side: int = 32,
        points_per_batch: int = 256,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.85,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 100,
        output_mode: Literal["multi_scale"] = "multi_scale",
        device: torch.device | str = "cuda",
    ):
        """
        Initialize the SAM1 model.

        Args:
            checkpoint: Model variant (vit_h, vit_l, vit_b).
            points_per_side: Grid density for point prompts.
            points_per_batch: Points processed simultaneously.
            pred_iou_thresh: Minimum predicted IoU for keeping a mask.
            stability_score_thresh: Minimum stability score for keeping a mask.
            stability_score_offset: Offset for stability score computation.
            mask_threshold: Binarization threshold for logit masks.
            box_nms_thresh: Box NMS IoU threshold within the image.
            min_mask_region_area: Min mask region area for predictor.
            output_mode: Only "multi_scale" is supported.
            device: Device to run on.
        """
        try:
            from segment_anything import SamPredictor, sam_model_registry
            import segment_anything.utils.amg as sam1_amg
        except ImportError:
            raise ImportError(_SAM1_INSTALL_MSG) from None

        self._amg = sam1_amg
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

        if checkpoint not in _SAM1_CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint '{checkpoint}'. Choose from: {list(_SAM1_CHECKPOINTS)}")

        registry_key, ckpt_name, url = _SAM1_CHECKPOINTS[checkpoint]
        self._logger.info(f"Loading SAM1 model ({checkpoint}, output_mode={output_mode})")

        path_to_weights = get_weights_path_for_model(ckpt_name, url, model_name="SAM1")
        sam = sam_model_registry[registry_key](checkpoint=str(path_to_weights))
        sam = sam.to(self._device)

        self._predictor = SamPredictor(sam)
        self._logger.info("SAM1 model loaded successfully.")

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
        Interface matches :meth:`SAM2Model.predict_masks_multi_scale`.

        Args:
            image: Image or crop, HWC uint8 (e.g. RGB).
            point_coords: Point prompts in image space, shape [N, 2] (x, y).
                If None, a single full-image grid is built using points_per_side.
            crop_box: [x0, y0, x1, y1] of this image in original frame. If None,
                image is treated as full image and crop_box = [0, 0, W, H].
            orig_size: (height, width) of the full image. If None, set to
                image.shape[:2]. Required when crop_box is provided.

        Returns:
            (masks_default, masks_s, masks_m, masks_l) -- lists of mask records.
        """
        amg = self._amg
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
        amg = self._amg
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

        self._predictor.reset_image()

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
        amg = self._amg
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
        """Run SAM1 on a batch of point prompts and split the 3 multimask outputs by index.

        Args:
            points: Point coordinates in image/crop space, shape [N, 2].
            im_size: (height, width) of the image/crop.
            crop_box: [x0, y0, x1, y1] in full-image frame.
            orig_size: (height, width) of the full image.

        Returns:
            (data_all, data_s, data_m, data_l) MaskData. Index 0 = small, 1 = medium, 2 = large.
        """
        amg = self._amg
        orig_h, orig_w = orig_size

        pts = torch.as_tensor(points, dtype=torch.float32, device=self._predictor.device)
        in_points = self._predictor.transform.apply_coords_torch(pts, im_size)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=pts.device)

        masks, iou_preds, _ = self._predictor.predict_torch(
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
