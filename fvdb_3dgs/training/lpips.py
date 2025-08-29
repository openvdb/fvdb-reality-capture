# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from enum import Enum
from typing import Literal, NamedTuple

import torch
import torchvision

from ..foundation_models.config import get_weights_path_for_model


def _get_tv_model_features(net: str, pretrained: bool = False) -> torch.nn.modules.container.Sequential:
    """
    Load neural net and pretrained features from torchvision by name.

    Currently supports squeezenet, alexnet, and vgg16 which are the backends for LPIPS.

    Args:
        net (str): Name of network. Must be one of "squeezenet1_1", "alexnet", or "vgg16".
        pretrained: If pretrained weights should be used or the network should be randomly initialized

    Return:
        nn.Module: The loaded network of the specified type

    >>> _ = _get_tv_model_features("alexnet", pretrained=True)
    >>> _ = _get_tv_model_features("squeezenet1_1", pretrained=True)
    >>> _ = _get_tv_model_features("vgg16", pretrained=True)

    """

    _weight_map = {
        "squeezenet1_1": "SqueezeNet1_1_Weights",
        "alexnet": "AlexNet_Weights",
        "vgg16": "VGG16_Weights",
    }

    if pretrained:
        model_weights = getattr(torchvision.models, _weight_map[net])
        model = getattr(torchvision.models, net)(weights=model_weights.DEFAULT)
    else:
        model = getattr(torchvision.models, net)(weights=None)
    return model.features


def _resize_tensor(x: torch.Tensor, size: int = 64) -> torch.Tensor:
    """
    Resize a batch of 2D tensors with shape (*, H, W) to (*, size, size) using torch.nn.functional.interpolate.

    Originally from:
        https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/sample_similarity_lpips.py#L127C22-L132.

    Args:
        x (torch.Tensor): The input tensor to resize of shape (*, H, W).
        size (int): The target size for the output tensor.

    Returns:
        torch.Tensor: The resized tensor with shape (*, size, size)
    """
    if x.shape[-1] > size and x.shape[-2] > size:
        return torch.nn.functional.interpolate(x, (size, size), mode="area")
    return torch.nn.functional.interpolate(x, (size, size), mode="bilinear", align_corners=False)


def _spatial_average(in_tens: torch.Tensor, keep_dim: bool = True) -> torch.Tensor:
    """
    Compute a spatial averaging over height and width of images.

    Args:
        in_tens (torch.Tensor): An image tensor of shape (B, C, H, W)
        keep_dim (bool): Whether to keep the spatial dimensions

    Returns:
        torch.Tensor: The spatially averaged tensor. If keep_dim is False, then the shape will be (B, C),
            otherwise, its shape will be (B, C, 1, 1)
    """
    return in_tens.mean([2, 3], keepdim=keep_dim)


def _upsample(in_tens: torch.Tensor, out_hw: tuple[int, ...] = (64, 64)) -> torch.Tensor:
    """
    Upsample an input image tensor (with shape (*, H, W)) with bilinear interpolation.

    Args:
        in_tens (torch.Tensor): A tensor of shape (*, H, W) to be resized
        out_hw (tuple[int, ...]): The target height and width for the output tensor.

    Returns:
        torch.Tensor: The resized tensor of shape (*, *out_hw)
    """
    return torch.nn.Upsample(size=out_hw, mode="bilinear", align_corners=False)(in_tens)


def _normalize_tensor(in_feat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize an input image tensor of shape (B, C, H, W) along its feature dimension (C).

    Args:
        in_feat (torch.Tensor): The input tensor of shape (B, C, H, W)
        eps (float): A small epsilon value to use in place of zero in sqrt

    Returns:
        torch.Tensor: The normalized tensor of shape (B, C, H, W)
    """
    norm_factor = torch.sqrt(eps + torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / norm_factor


def _valid_img(img: torch.Tensor, normalize: bool) -> bool:
    """
    Check that input is a valid image to the network. i.e. has the right shape, and has values in
    the right range ([0, 1] if normalize is True)

    Args:
        img (torch.Tensor): The input tensor of shape (B, C, H, W)
        normalize (bool): Whether the input tensor is normalized

    Returns:
        bool: True if the input tensor is valid, False otherwise
    """

    value_check = img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    return img.ndim == 4 and img.shape[1] == 3 and value_check  # type: ignore[return-value]


class SqueezeNet(torch.nn.Module):
    """
    Implementation of SqueezeNet compatible with torchvision.models.SqueezeNet1_1_Weights weights.

    Instead of returning classification labels, returns intermediate features after each block of layers.
    """

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        """
        Initialize a SqueezeNet module with optional gradient tracking and pretrained weights.

        Args:
            requires_grad (bool): Whether the weights of the network need to track gradients for training. Default is False.
            pretrained (bool): Whether to load pretrained weights from torchvision.models. Defaults is True.
        """
        super().__init__()
        pretrained_features = _get_tv_model_features("squeezenet1_1", pretrained)

        self.N_slices = 7
        slices = []
        feature_ranges = [range(2), range(2, 5), range(5, 8), range(8, 10), range(10, 11), range(11, 12), range(12, 13)]
        for feature_range in feature_ranges:
            seq = torch.nn.Sequential()
            for i in feature_range:
                seq.add_module(str(i), pretrained_features[i])
            slices.append(seq)

        self.slices = torch.nn.ModuleList(slices)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> NamedTuple:
        """
        Call the SqueezeNet forward pass and return features after each block of layers.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) representing a batch of images.

        Returns:
            _SqueezeOutput: A named tuple containing the output features after each block of layers.
                There are seven feature layers named `relu1`, ..., `relu7` (because they are the output of a ReLU layer).
                Each has shape (B, C, H_i, W_i) for i = 1, ..., 7
        """

        class _SqueezeOutput(NamedTuple):
            relu1: torch.Tensor
            relu2: torch.Tensor
            relu3: torch.Tensor
            relu4: torch.Tensor
            relu5: torch.Tensor
            relu6: torch.Tensor
            relu7: torch.Tensor

        relus = []
        for slice_ in self.slices:
            x = slice_(x)
            relus.append(x)
        return _SqueezeOutput(*relus)


class AlexNet(torch.nn.Module):
    """
    Implementation of AlexNet compatible with torchvision.models.AlexNet_Weights weights.

    Instead of returning classification labels, returns intermediate features after each block of layers.
    """

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        """
        Initialize an AlexNet module with optional gradient tracking and pretrained weights.

        Args:
            requires_grad (bool): Whether the weights of the network need to track gradients for training. Default is False.
            pretrained (bool): Whether to load pretrained weights from torchvision.models. Defaults is True.
        """
        super().__init__()
        alexnet_pretrained_features = _get_tv_model_features("alexnet", pretrained)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> NamedTuple:
        """
        Call the AlexNet forward pass and return features after each block of layers.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) representing a batch of images.

        Returns:
            _AlexnetOutputs: A named tuple containing the output features after each block of layers.
                There are five feature layers named `relu1`, ..., `relu5` (because they are the output of a ReLU layer).
                Each has shape (B, C, H_i, W_i) for i = 1, ..., 5
        """
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h

        class _AlexnetOutputs(NamedTuple):
            relu1: torch.Tensor
            relu2: torch.Tensor
            relu3: torch.Tensor
            relu4: torch.Tensor
            relu5: torch.Tensor

        return _AlexnetOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class Vgg16(torch.nn.Module):
    """
    Implementation of Vgg16 compatible with torchvision.models.VGG16_Weights weights.

    Instead of returning classification labels, returns intermediate features after each block of layers.
    """

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        """
        Initialize a Vgg16 module with optional gradient tracking and pretrained weights.

        Args:
            requires_grad (bool): Whether the weights of the network need to track gradients for training. Default is False.
            pretrained (bool): Whether to load pretrained weights from torchvision.models. Defaults is True.
        """
        super().__init__()
        vgg_pretrained_features = _get_tv_model_features("vgg16", pretrained)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> NamedTuple:
        """
        Call the Vgg16 forward pass and return features after each block of layers.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) representing a batch of images.

        Returns:
            _VGGOutputs: A named tuple containing the output features after each block of layers.
                There are five feature layers named `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`, `relu5_3`
                (because they are the output of a ReLU layer). Each has shape (B, C, H_i, W_i) for i = 1, ..., 5
        """
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        class _VGGOutputs(NamedTuple):
            relu1_2: torch.Tensor
            relu2_2: torch.Tensor
            relu3_3: torch.Tensor
            relu4_3: torch.Tensor
            relu5_3: torch.Tensor

        return _VGGOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class ImageWhiteningLayer(torch.nn.Module):
    """
    A layer which whitens images according the mean and variance of image pixels in ImageNet.

    i.e. applies input = (input - shift) / scale
    where shift = [-0.030, -0.088, -0.188] and scale = [0.458, 0.448, 0.450]
    which are the mean pixel color and variance of pixel colors over the ImageNet dataset.
    """

    shift: torch.Tensor
    scale: torch.Tensor

    def __init__(self) -> None:
        """
        Create a new ImageWhiteningLayer layer.
        """
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None], persistent=False)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Whiten an input image by shifting and scaling by the mean and variance of pixel colors in ImageNet.

        Args:
            inp (torch.Tensor): An input tensor of shape (*, 3)

        Returns:
            torch.Tensor: The whitened input tensor.
        """
        return (inp - self.shift) / self.scale


class LinearLayerWithDropout(torch.nn.Module):
    """
    A single linear layer implemented as a 1x1 conv with optional dropout.
    Equivalent to W Dropout(x) + b or Wx + b.
    """

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        """
        Initialize a new Linear layer mapping from chn_in dimensions to chn_out dimensions with optional dropout.

        Args:
            chn_in (int): The number of input features of the layer.
            chn_out (int): The number of output features of the layer.
            use_dropout (bool): Whether to apply dropout to the input. Defaults to False.
        """
        super().__init__()

        layers = [torch.nn.Dropout()] if use_dropout else []
        layers += [
            torch.nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),  # type: ignore[list-item]
        ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear layer to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (*, chn_in)

        Returns:
            torch.Tensor: The output of the linear layer of shape (*, chn_out)
        """
        return self.model(x)


class LPIPSNetwork(torch.nn.Module):
    """
    Implementation of the LPIPS loss network as an nn.Module.

    LPIPS works in two phases:
        1. First it computes the squared distances between activations of a reference and target
        image using a pretrained image classification backbone network.
        __i.e.__ given images, img1, img2 both with shape (B, 3, H, W), it computes a stack of
        distances d1, ..., dL of shape (B, C_i, H_i, W_i) which are the difference in activations
        of a CNN image classification backbone.
        2. The differences are each fed through a linear layer followed by an resampling layer
        which projects them to a tensor of
        shape (B, 1, H, W) where C is the number of output channels, and the results are summed together
        to form the final score. The linear layers are pretrained to score the similarity between features.
    """

    def __init__(
        self,
        pretrained: bool = True,
        backbone: Literal["alex", "vgg", "squeeze"] = "alex",
        spatial_average_features: bool = True,
        use_pretrained_backbone: bool = True,
        enable_backprop: bool = False,
        use_dropout: bool = True,
        eval_mode: bool = True,
        resize: int | None = None,
    ) -> None:
        """
        Create a new LPIPSNetwork network for measuring the similarity between images using the specified
        image classifier backbone (by default using models pretrained on ImageNet).

        Args:
            pretrained: If True, load pretrained weights for the linear layers which compute the pixel-wise
                similarity between image layers. Otherwise, use a random initialization (useful e.g. if you
                wanted to train your own LPIPSNetwork). Defaults to True.
            backbone: Indicate which backbone to use, choose between ['alex','vgg','squeeze'] represengint
                AlexNet, VGG16, and SqueezeNet respectively. Defaults to 'alex'.
            spatial_average_features: If the outputs of backbone layers should be spatially averaged across the image. Defaults to True
            use_pretrained_backbone: If backbone should be random or use imagenet pre-trained weights. Default is True.
            enable_backprop: If backprop should be enabled for both backbone and linear layers
                (useful if you want to use this as a loss during training). Default is alse.
            use_dropout: If dropout layers should be added to the linear layers.
            eval_mode: If network should be in evaluation mode (i.e. will not update batchnorm or apply dropout, etc.).
            resize: If input images should be rescaled to resize x resize before passing to the network. If None, uses the original size.

        """
        super().__init__()

        self._backbone_type = backbone
        self._enable_backprop = enable_backprop
        self._init_backbone_random = not use_pretrained_backbone
        self._spatial_average_backbone_features = spatial_average_features
        self._input_rescale_size = resize
        self._scaling_layer = ImageWhiteningLayer()

        if self._backbone_type in ["vgg", "vgg16"]:
            net_type = Vgg16
            self._backbone_channels = [64, 128, 256, 512, 512]
        elif self._backbone_type == "alex":
            net_type = AlexNet
            self._backbone_channels = [64, 192, 384, 256, 256]
        elif self._backbone_type == "squeeze":
            net_type = SqueezeNet
            self._backbone_channels = [64, 128, 256, 384, 384, 512, 512]
        self._num_backbone_layers = len(self._backbone_channels)

        self.net = net_type(pretrained=not self._init_backbone_random, requires_grad=self._enable_backprop)

        self.lin0 = LinearLayerWithDropout(self._backbone_channels[0], use_dropout=use_dropout)
        self.lin1 = LinearLayerWithDropout(self._backbone_channels[1], use_dropout=use_dropout)
        self.lin2 = LinearLayerWithDropout(self._backbone_channels[2], use_dropout=use_dropout)
        self.lin3 = LinearLayerWithDropout(self._backbone_channels[3], use_dropout=use_dropout)
        self.lin4 = LinearLayerWithDropout(self._backbone_channels[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self._backbone_type == "squeeze":  # 7 layers for squeezenet
            self.lin5 = LinearLayerWithDropout(self._backbone_channels[5], use_dropout=use_dropout)
            self.lin6 = LinearLayerWithDropout(self._backbone_channels[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]
        self.lins = torch.nn.ModuleList(self.lins)

        if pretrained:
            weights_url = f"https://www.fwilliams.info/files/lpips_models/{backbone}.pth"
            path_to_weights = get_weights_path_for_model(f"{backbone}.pth", weights_url, model_name=backbone)
            self.load_state_dict(torch.load(path_to_weights, map_location="cpu", weights_only=False), strict=False)

        if eval_mode:
            self.eval()

        if not self._enable_backprop:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self, in0: torch.Tensor, in1: torch.Tensor, retperlayer: bool = False, normalize: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the LPIPSNetwork.

        Args:
            in0 (torch.Tensor): First input tensor.
            in1 (torch.Tensor): Second input tensor.
            retperlayer (bool): Whether to return per-layer outputs.
            normalize (bool): Whether to resacale inputs to [-1, 1] from [0, 1]

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: LPIPS score or per-layer outputs.
        """
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # normalize input
        in0_input, in1_input = self._scaling_layer(in0), self._scaling_layer(in1)

        # resize input if needed
        if self._input_rescale_size is not None:
            in0_input = _resize_tensor(in0_input, size=self._input_rescale_size)
            in1_input = _resize_tensor(in1_input, size=self._input_rescale_size)

        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self._num_backbone_layers):
            feats0[kk], feats1[kk] = _normalize_tensor(outs0[kk]), _normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = []
        for kk in range(self._num_backbone_layers):
            if self._spatial_average_backbone_features:
                res.append(_spatial_average(self.lins[kk](diffs[kk]), keep_dim=True))
            else:
                res.append(_upsample(self.lins[kk](diffs[kk]), out_hw=tuple(in0.shape[2:])))

        val: torch.Tensor = sum(res)  # type: ignore[assignment]
        if retperlayer:
            return (val, res)
        return val


class LPIPSNetworkNoTrain(LPIPSNetwork):
    """
    Wrapper around LPIPSNetwork to make sure it never leaves evaluation mode.
    """

    def train(self, mode: bool) -> "LPIPSNetworkNoTrain":
        """
        Overload of train() method which ignores the mode and forces the network to always be in evaluation mode.

        Args:
            mode (bool): Ignored but there for API compatibility with super class

        Returns:
            _NoTrainLpips: The network wrapped in a no-train wrapper.
        """
        return super().train(False)


class LPIPSLoss(torch.nn.Module):
    """
    The Learned Perceptual Image Patch Similarity (LPIPS) calculates perceptual similarity between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(B, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `backbone` arg).
    """

    def __init__(
        self,
        backbone: Literal["vgg", "alex", "squeeze"] = "alex",
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = True,
        enable_backprop: bool = False,
    ) -> None:
        """
        Initialize a new LPIPSLoss.

        Args:
            backbone (Literal['alex', 'squeeze', 'vgg']): Which backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
            reduction (Literal['sum', 'mean']): How to reduce over the batch dimension. Choose between `'sum'` or `'mean'`.
            normalize (bool): Whether to rescale the inputs to [-1, 1] from [0, 1]. Default is True and the loss
                expects inputs in the range [0, 1]. Set to False if input images are expected in [-1, 1]
            enable_backprop (bool): Whether to enable backpropagation through this loss function. Default is False.
        """
        super().__init__()
        valid_net_type = ("vgg", "alex", "squeeze")
        if backbone not in valid_net_type:
            raise ValueError(f"Argument `net_type` must be one of {valid_net_type}, but got {backbone}.")
        self.net = LPIPSNetworkNoTrain(backbone=backbone, enable_backprop=enable_backprop)

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        self.reduction: Literal["mean", "sum"] = reduction

        if not isinstance(normalize, bool):
            raise ValueError(f"Argument `normalize` should be an bool but got {normalize}")
        self.normalize = normalize

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the LPIPS loss between two images of shape (B, 3, H, W).

        Args:
            img1 (torch.Tensor): The first image of shape (B, 3, H, W)
            img2 (torch.Tensor): The second image of shape (B, 3, H, W)

        Returns:
            torch.Tensor: The LPIPS loss between the two images
        """
        if not (_valid_img(img1, self.normalize) and _valid_img(img2, self.normalize)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors with shape [B, 3, H, W]."
                f" Got input with shape {img1.shape} and {img2.shape} and values in range"
                f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
                f" expected to be in the {[0, 1] if self.normalize else [-1, 1]} range."
            )
        loss = self.net(img1, img2, normalize=self.normalize).squeeze()
        batch_size = img1.shape[0]
        return loss / batch_size if self.reduction == "mean" else loss
