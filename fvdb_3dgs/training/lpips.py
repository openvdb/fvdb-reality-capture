# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Literal, NamedTuple

import torch
import torchvision

from ..foundation_models.config import get_weights_path_for_model

# This file is modified from code in the TorchMetrics library (licensed under Apache v2).
# The torchmetrics license is included here:
# Namely, the torch.nn.Module definitions for SqueezeNet, AlexNet, and Vgg16 are lightly
# modified from torchmetrics.
# Original source here https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/image/lpips.py
# The LPIPS torch module is heavily modified from its torchmetrics version.
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright 2020-2022 Lightning-AI team
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


def _get_tv_model_features(net: str, pretrained: bool = False) -> torch.nn.modules.container.Sequential:
    """
    Load neural net and pretrained features from torchvision by name.

    Currently supports squeezenet, alexnet, and vgg16 which are the backends for LPIPS.

    Args:
        net (str): Name of network. Must be one of "squeezenet1_1", "alexnet", or "vgg16".
        pretrained: If pretrained weights should be used or the network should be randomly initialized

    Return:
        nn.Module: The loaded network of the specified type

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
            weights_url = f"https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/{backbone}.pth"
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
