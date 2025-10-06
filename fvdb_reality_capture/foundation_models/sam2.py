# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Literal

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from .config import get_weights_path_for_model


class SAM2Model:
    """
    A wrapper for the SAM2 (Segment Anything Model 2) model for segmentation.  Used for evaluation only.

    SAM2 is a foundation model solving promptable visual segmentation in images and videos.

    The original SAM2 paper can be found here:
        https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        checkpoint: Literal["large", "small", "tiny", "base_plus"] = "large",
        points_per_side: int = 40,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.80,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize a SAM2 model for evaluation.

        Args:
            checkpoint (Literal["large", "small", "tiny", "base_plus"]): Checkpoint to use for the SAM2 model.
            points_per_side (int): Defines a grid of evenly spaced points for point prompts of the SAM2 model. For example, if points_per_side=32, the model places a grid of 32x32 points over the image. Defaults to 40.
            pred_iou_thresh (float): The minimum predicted IoU score for a mask to be included in the output. Defaults to 0.80.
            stability_score_thresh (float): The minimum stability score for a mask to be included in the output. Defaults to 0.80.
            device (torch.device | str): Device to load the model on (default is "cuda").
        """

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._logger.info(f"Loading SAM2 model with {checkpoint} checkpoint")

        # Download the checkpoint
        ckpt_name = f"sam2.1_hiera_{checkpoint}.pt"
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ckpt_name}"
        path_to_weights = get_weights_path_for_model(ckpt_name, url, model_name="SAM2")

        # Get the model config name
        config_name_map = {
            "large": "l",
            "small": "s",
            "tiny": "t",
            "base_plus": "b+",
        }
        config_name = config_name_map[checkpoint]
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{config_name}"

        # Build the model
        device = torch.device(device)
        sam2_model = build_sam2(model_cfg, ckpt_path=path_to_weights, device=str(device), apply_postprocessing=False)
        self._sam2_mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

        self._logger.info("SAM2 model loaded successfully.")

    def predict_masks(
        self,
        image: np.ndarray,
    ):
        """
        Predict masks for an image using the SAM2 model.

        Args:
            image (torch.Tensor): Image to predict masks for, shape [H, W, C], uint8 format.
        Returns:
           list(dict(str, any)): A list over records for masks. Each record is a dict containing the
            following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If output_mode='binary_mask',
                    is an array of shape HW. Otherwise, is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's quality. This is
                    filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input to the model to
                    generate this mask.
               stability_score (float): A measure of the mask's quality. This is filtered on using
                    the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate the mask, given in
                    XYWH format.
        """

        return self._sam2_mask_generator.generate(image)
