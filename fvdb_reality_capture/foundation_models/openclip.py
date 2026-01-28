# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""OpenCLIP model wrapper for image and text encoding.

This module provides a simple wrapper around the OpenCLIP model for encoding
images and text into a shared embedding space.

The original OpenCLIP project can be found here:
    https://github.com/mlfoundations/open_clip
"""
import logging

import torch
import torchvision.transforms as T

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip is not installed. Install it with: pip install open-clip-torch")


# Mapping from torch dtype to OpenCLIP precision string
_DTYPE_TO_PRECISION = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


class OpenCLIPModel:
    """
    A simple wrapper for the OpenCLIP model for encoding images and text.

    This provides basic functionality to encode images and text into CLIP's
    shared embedding space. For application-specific logic (e.g., relevancy
    computation with positive/negative prompts), create a wrapper class that
    uses this model.

    Example usage:

    .. code-block:: python

        from fvdb_reality_capture.foundation_models import OpenCLIPModel

        model = OpenCLIPModel(device="cuda")

        # Encode images (expects tensor with shape [B, C, H, W] or [C, H, W])
        image_features = model.encode_image(images)

        # Encode text
        text_features = model.encode_text(["a photo of a cat", "a photo of a dog"])
    """

    def __init__(
        self,
        model_type: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize the OpenCLIP model.

        Args:
            model_type: CLIP model architecture (e.g., 'ViT-B-16', 'ViT-L-14', 'ViT-H-14').
            pretrained: Pretrained weights identifier (e.g., 'laion2b_s34b_b88k', 'openai').
            dtype: Model dtype - torch.float32, torch.float16, or torch.bfloat16. Defaults to torch.float16.
            device: Device to run the model on.
        """
        super().__init__()

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._device = torch.device(device)
        self._model_type = model_type
        self._pretrained = pretrained
        self._dtype = dtype

        # Convert torch dtype to OpenCLIP precision string
        if dtype not in _DTYPE_TO_PRECISION:
            raise ValueError(f"Unsupported dtype: {dtype}. " f"Supported dtypes: {list(_DTYPE_TO_PRECISION.keys())}")
        precision_str = _DTYPE_TO_PRECISION[dtype]

        self._logger.info(f"Loading OpenCLIP model: {model_type} pretrained on {pretrained} (dtype: {dtype})")

        # Load the CLIP model and its associated preprocessing transform
        # The preprocess transform is specific to the model and pretrained weights,
        # including the correct image size and normalization parameters
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_type,
            pretrained=pretrained,
            precision=precision_str,
        )
        model.eval()

        self._model = model.to(self._device)
        self._tokenizer = open_clip.get_tokenizer(model_type)
        self._preprocess = preprocess

        # Get embedding dimension from model
        self._embedding_dim = self._model.visual.output_dim

        # Get the expected image size from the model
        # This is typically 224, but can be 336, 384, etc. for some models
        self._image_size = self._model.visual.image_size
        if isinstance(self._image_size, (list, tuple)):
            self._image_size = self._image_size[0]  # square

        # Extract normalization parameters from the PIL preprocess transform
        norm_params = self._extract_normalize_params(preprocess)
        if norm_params is not None:
            self._normalize_mean, self._normalize_std = norm_params
        else:
            # No normalization found - use identity (no-op)
            self._normalize_mean = (0.0, 0.0, 0.0)
            self._normalize_std = (1.0, 1.0, 1.0)
            self._logger.warning(
                "No Normalize transform found in preprocess - tensor preprocessing will skip normalization"
            )

        # Create a tensor-compatible preprocessing transform
        # This can be used with tensors that have values in [0, 1] range
        tensor_transforms = [T.Resize((self._image_size, self._image_size), antialias=True)]
        if norm_params is not None:
            tensor_transforms.append(T.Normalize(mean=self._normalize_mean, std=self._normalize_std))
        self._preprocess_tensor = T.Compose(tensor_transforms)

        self._logger.info(
            f"OpenCLIP model loaded successfully. "
            f"Embedding dim: {self._embedding_dim}, Image size: {self._image_size}"
        )

    @staticmethod
    def _extract_normalize_params(preprocess) -> tuple[tuple, tuple] | None:
        """Extract mean and std from the Normalize step in a Compose transform.

        Args:
            preprocess: A torchvision.transforms.Compose object (as returned by OpenCLIP).

        Returns:
            Tuple of (mean, std) extracted from the Normalize transform,
            or None if no Normalize transform is found.
        """
        if hasattr(preprocess, "transforms"):
            for transform in preprocess.transforms:
                if isinstance(transform, T.Normalize):
                    return tuple(transform.mean), tuple(transform.std)
        return None

    @property
    def model_type(self) -> str:
        """Return the model architecture type."""
        return self._model_type

    @property
    def pretrained(self) -> str:
        """Return the pretrained weights identifier."""
        return self._pretrained

    @property
    def dtype(self) -> torch.dtype:
        """Return the model dtype."""
        return self._dtype

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embedding_dim

    @property
    def image_size(self) -> int:
        """Return the expected input image size (assumes square images)."""
        return self._image_size

    @property
    def device(self) -> torch.device:
        """Return the current device."""
        return self._device

    @property
    def normalize_mean(self) -> tuple:
        """Return the normalization mean values (R, G, B)."""
        return self._normalize_mean

    @property
    def normalize_std(self) -> tuple:
        """Return the normalization std values (R, G, B)."""
        return self._normalize_std

    @property
    def preprocess(self):
        """Return the preprocessing transform for PIL images.

        This is a torchvision.transforms.Compose that expects PIL images
        as input and returns a normalized tensor. It includes:
        - Resize to model's expected image size
        - CenterCrop (for some models)
        - ToTensor (converts PIL image to tensor)
        - Normalize with CLIP mean/std

        For inputs that are already tensors, use :attr:`preprocess_tensor` instead.
        """
        return self._preprocess

    @property
    def preprocess_tensor(self):
        """Return a tensor-compatible preprocessing transform.

        This transform expects tensors with shape [C, H, W] or [B, C, H, W]
        and values in [0, 1] range. It applies:
        - Resize to model's expected image size
        - Normalize with CLIP mean/std

        Example:
            >>> images = torch.rand(4, 3, 256, 256)  # values in [0, 1]
            >>> preprocessed = model.preprocess_tensor(images)
            >>> embeddings = model.encode_image(preprocessed)
        """
        return self._preprocess_tensor

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images into CLIP feature space.

        Args:
            images: Preprocessed images as tensor, shape [B, C, H, W] or [C, H, W].
                    Should already be preprocessed using self.preprocess or
                    normalized with values typically in [-2, 2] range after
                    the model's normalization.

        Returns:
            CLIP embeddings, shape [B, embedding_dim].
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)

        images = images.to(dtype=self._dtype, device=self._device)
        embeddings = self._model.encode_image(images)
        return embeddings

    @torch.no_grad()
    def encode_text(self, text_list: list[str]) -> torch.Tensor:
        """
        Encode text prompts into CLIP feature space.

        Args:
            text_list: List of text prompts to encode.

        Returns:
            CLIP embeddings, shape [len(text_list), embedding_dim].
        """
        if not text_list:
            raise ValueError("encode_text expected a non-empty text_list")
        tokens = self._tokenizer(text_list).to(self._device)
        embeddings = self._model.encode_text(tokens)
        return embeddings

    def to(self, device: torch.device | str) -> "OpenCLIPModel":
        """Move the model to a different device."""
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
