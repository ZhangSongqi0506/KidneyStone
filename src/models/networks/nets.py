import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from pydoc import cli
from traceback import print_tb
from grpc import ClientCallDetails
import torch
from torch import nn
# from torchmtlr import MTLR
import torch.nn.functional as F
from typing import Sequence, Tuple, Union

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
# from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from einops import repeat, rearrange

import sys

# Number of clin_var
n_clin_var = 15


def flatten_layers(arr):
    return [i for sub in arr for i in sub]


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Union[Sequence[int], int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "conv",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            patch_size: int = 16
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        config = {}
        config['num_of_attention_heads'] = 12
        config['hidden_size'] = 768
        # self.msa = BertSelfAttention(config)
        self.mtlr = nn.Linear(hidden_size, 1)
        # if hparams['n_dense'] <= 0:
        #     # self.mtlr = MTLR(hparams['hidden_size'], hparams['time_bins'])
        #     self.mtlr = nn.Linear(hparams['hidden_size'], hparams['time_bins'])
        #
        # else:
        #     fc_layers = [[nn.Linear(hparams['hidden_size'], 256 * hparams['dense_factor']),
        #                   nn.BatchNorm1d(256 * hparams['dense_factor']),
        #                   nn.ReLU(inplace=True),
        #                   nn.Dropout(hparams['dropout'])]]
        #
        #     if hparams['n_dense'] > 1:
        #         fc_layers.extend([[nn.Linear(256 * hparams['dense_factor'], 64 * hparams['dense_factor']),
        #                            nn.BatchNorm1d(64 * hparams['dense_factor']),
        #                            nn.ReLU(inplace=True),
        #                            nn.Dropout(hparams['dropout'])] for _ in range(hparams['n_dense'] - 1)])
        #
        #     fc_layers = flatten_layers(fc_layers)
        #     # self.mtlr = nn.Sequential(*fc_layers,
        #     #                           MTLR(64 * hparams['dense_factor'], hparams['time_bins']), )
        #     self.mtlr = nn.Sequential(*fc_layers,
        #                               nn.Linear(64 * hparams['dense_factor'], hparams['time_bins']), )

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, sample):

        sample_img, clin_var = sample
        x_in = (sample_img, clin_var)

        x, hidden_states_out = self.vit(x_in)

        enc1 = self.encoder1(sample_img)
        x2 = hidden_states_out[3][:, 1:, :]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6][:, 1:, :]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9][:, 1:, :]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x[:, 1:, :], self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        x = torch.mean(x, dim=1)
        risk_out = self.mtlr(x)

        return self.out(out), risk_out

    # Copyright 2020 - 2021 MONAI Consortium


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock


# __all__ = ["ViT"]


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            pos_embed: str = "conv",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())

        ## Projection of EHR
        self.EHR_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):

        x = self.patch_embedding(x)  # img, clin_var = x

        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if self.classification:
            x = self.classification_head(x[:, 0])

        return x, hidden_states_out

    # Copyright 2020 - 2021 MONAI Consortium


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}


class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4, pos_embed="conv")

    """

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int,
            num_heads: int,
            pos_embed: str,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = look_up_option(pos_embed, SUPPORTED_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)]) + 1  # +1 for EHR
        self.patch_dim = in_channels * np.prod(patch_size)

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i + 1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )

        self.EHR_proj = nn.Sequential(nn.Linear(n_clin_var, hidden_size))

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        # From PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def forward(self, x):
        img, clin_var = x
        x = self.patch_embeddings(img)

        # print(x.shape)

        clin_var = self.EHR_proj(clin_var)
        # print(clin_var.shape)
        clin_var = clin_var.unsqueeze(dim=1)
        # print(clin_var.shape)


        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
            # print(x.shape)

        x = torch.cat([clin_var, x], dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class ViTNoEmbed(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            pos_embed: str = "conv",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())

    def forward(self, x):

        # x = self.patch_embedding(x)

        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if self.classification:
            x = self.classification_head(x[:, 0])

        return x, hidden_states_out
from monai.networks.blocks import PatchEmbeddingBlock as PatchEmbeddingBlockOriginal
class DoubleFlow(nn.Module):
    def __init__(
                    self,
                    in_channels: int,
                    out_channels: int,
                    img_size: Union[Sequence[int], int],
                    feature_size: int = 16,
                    hidden_size: int = 768,
                    mlp_dim: int = 3072,
                    num_heads: int = 12,
                    pos_embed: str = "conv",
                    norm_name: Union[Tuple, str] = "instance",
                    conv_block: bool = True,
                    res_block: bool = True,
                    dropout_rate: float = 0.0,
                    spatial_dims: int = 3,
                    patch_size: int = 8
                ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size

        self.classification = False

        self.img_patch_embedding = PatchEmbeddingBlockOriginal(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )


        self.vit = ViTNoEmbed(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(256, 1))

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x):
        img_in, clin_in = x

        # Image path
        img = self.img_patch_embedding(img_in) # 216*h
        img_clin = self.patch_embedding(x) # 217*h
        # clinical and img out the end layer : x
        x, _ = self.vit(img_clin)
        # img out all layers: hidden_states_out
        _, hidden_states_out = self.vit(img)

        enc1 = self.encoder1(img_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))

        dec4 = self.proj_feat(_, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        # print(f'img_in:{img_in.shape}\nz3{x2.shape}\nz6{x3.shape}\nz9{x4.shape}\nz12{_.shape}')
        #
        # for i in [enc1, enc2, enc3, enc4]:
        #     print(i.shape)
        # for i in [dec4, dec3, dec2, dec1]:
        #     print(i.shape)


        segmentation_output = self.decoder2(dec1, enc1)

        x = torch.mean(x, dim=1)
        classification_output = self.fc(x)

        return self.out(segmentation_output), classification_output



class ehr_net(torch.nn.Module):
    def __init__(self):
        super(ehr_net, self).__init__()
        # self.EHR_proj = nn.Sequential(nn.Linear(n_clin_var, hidden_size))
        self.fc = nn.Sequential(nn.Linear(15, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(256, 1))
    def forward(self, clin_var):
        clin_var = self.fc(clin_var)
        return clin_var




class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        # self.img_embed = PatchEmbeddingBlockOriginal(
        #     in_channels=1, img_size=48, patch_size=16, hidden_size=hidden_size, num_heads=12)
        #
        # self.ehr_proj = nn.Linear(15, hidden_size).unsqueeze(0)
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v):
        # 图像嵌入处理，作为键和值
        # img_embedded = self.img_embed(img)  # 假设img_embed返回的是NxD的向量，N为图像补丁数量，D为隐藏层大小
        keys = self.key_projection(k)
        values = self.value_projection(v)
        # EHR数据处理，作为查询
        # ehr_projected = self.ehr_proj(ehr)  # 假设ehr为1x15的向量
        query = self.query_projection(q)  # 增加批处理维度以匹配keys和values的维度
        # 计算注意力分数
        attention_scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        # 归一化注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        # 计算上下文向量
        context = torch.matmul(attention_weights, values)
        return context


def get_fusion_model(hidden_size):

    fusion_model = nn.Sequential(
        nn.Linear(in_features=hidden_size, out_features=256),  # 从384到256
        nn.BatchNorm1d(256),  # 加入BatchNorm稳定训练
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(in_features=256, out_features=128),  # 从256到128
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(in_features=128, out_features=32),  # 从128到32
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),  # 最后一层Dropout可以减少
        nn.Linear(in_features=32, out_features=1),  # 输出单个值
    )
    return fusion_model


class HyMNet(nn.Module):
    def __init__(self,
                 img_size: Union[Sequence[int], int],
                 in_channels: int = 1,
                 out_channels: int = 1,
                 feature_size: int = 16,
                 patch_size: int = 16,
                 hidden_size: int = 384,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 dropout_rate: float = 0.0,
                 spatial_dims: int=3,
                 norm_name: Union[Tuple, str] = "instance",
                 conv_block: bool = True,
                 res_block: bool = True,
                 ):

        super().__init__()

        self.fusion_model = get_fusion_model(hidden_size)
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size

        # CT embedding and projection
        self.img_embed = PatchEmbeddingBlockOriginal(
            in_channels=in_channels, img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, num_heads=num_heads)
        # ehr projection
        self.ehr_proj = nn.Linear(15, hidden_size)

        self.cross_attention = CrossAttention(hidden_size=hidden_size)

        self.vit = ViTNoEmbed(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            classification=False,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        # seg out
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        # cls out
        self.fc = nn.Sequential(nn.Linear(hidden_size * 1, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(256, 1))


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, img, ehr):
        # 图像嵌入
        img_embeded = self.img_embed(img)
        # EHR投影，并增加一个维度
        ehr_proj = self.ehr_proj(ehr).unsqueeze(dim=1)

        # # 图像与EHR的注意力机制
        # # context_img_ehr = self.cross_attention(q=ehr_proj, k=img_embeded, v=img_embeded)  # 1*hidden_size 不好
        # context_img_ehr = self.cross_attention(q=img_embeded, k=ehr_proj, v=ehr_proj)  # 1*hidden_size
        # 分割路径
        _, hidden_states_out = self.vit(img_embeded)

        # 编码器1
        enc1 = self.encoder1(img)
        # 从ViT模型中获取的第3层隐藏状态
        x2 = hidden_states_out[3]
        # 编码器2
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        # 获取ViT的第6层隐藏状态
        x3 = hidden_states_out[6]
        # 编码器3
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        # 获取ViT的第9层隐藏状态
        x4 = hidden_states_out[9]
        # 编码器4
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))

        # 解码器4的输入
        dec4 = self.proj_feat(_, self.hidden_size, self.feat_size)
        # 解码器5
        dec3 = self.decoder5(dec4, enc4)
        # 解码器4
        dec2 = self.decoder4(dec3, enc3)
        # 解码器3
        dec1 = self.decoder3(dec2, enc2)
        # 解码器2
        dec0 = self.decoder2(dec1, enc1)
        # 输出分割结果
        seg_out = self.out(dec0)
        # 从ViT最后一层隐藏状态获取平均值
        z12_mean = torch.mean(hidden_states_out[-1], dim=1).unsqueeze(dim=1)
        # print(context_img_ehr.shape)
        # print(img_embeded.shape)
        # print(ehr_proj.shape)
        combined_embeddings = torch.cat((z12_mean, ehr_proj), 1)
        # combined_embeddings = torch.cat((img_embeded, ehr_proj), 1)
        combined_embeddings = torch.mean(combined_embeddings, dim=1)
        # print(combined_embeddings.shape)


        cls_output = self.fusion_model(combined_embeddings)

        # # 分割-注意力机制
        # # context_seg_img_ehr = z12_mean  # 只有临床数据
        # # context_seg_img_ehr = context_img_ehr  # 丢弃SMA
        # context_seg_img_ehr = self.cross_attention(q=z12_mean, k=context_img_ehr, v=context_img_ehr)
        # context_seg_img_ehr = torch.mean(context_seg_img_ehr, dim=1)
        # print(z12_mean.shape, z12_mean.flatten(start_dim=1).shape, context_img_ehr.shape)
        #
        # # context_seg_img_ehr = torch.cat((z12_mean, context_img_ehr), dim=1)  # 仅临床和CT
        # # context_seg_img_ehr = context_img_ehr
        # print(context_seg_img_ehr.shape)
        # # 全连接层，生成分类输出
        # cls_output = self.fc(context_seg_img_ehr.flatten(start_dim=1))
        return seg_out, cls_output
class KSCNet(nn.Module):
    def __init__(self,
                 img_size: Union[Sequence[int], int],
                 in_channels: int = 1,
                 out_channels: int = 1,
                 feature_size: int = 16,
                 patch_size: int = 16,
                 hidden_size: int = 384,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 dropout_rate: float = 0.0,
                 spatial_dims: int=3,
                 pos_embed: str = "conv",
                 norm_name: Union[Tuple, str] = "instance",
                 conv_block: bool = True,
                 res_block: bool = True,
                 ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.fusion_model = get_fusion_model(hidden_size)
        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size

        # CT embedding and projection
        self.img_embed = PatchEmbeddingBlockOriginal(
            in_channels=in_channels, img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, num_heads=num_heads,pos_embed=pos_embed)
        # ehr projection
        self.ehr_proj = nn.Linear(15, hidden_size)

        self.cross_attention = CrossAttention(hidden_size=hidden_size)

        self.vit = ViTNoEmbed(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            classification=False,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        # seg out
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        # cls out
        self.fc = nn.Sequential(nn.Linear(hidden_size * 1, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(256, 1))


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, img, ehr):
        # 图像嵌入
        img_embeded = self.img_embed(img)
        # EHR投影，并增加一个维度
        ehr_proj = self.ehr_proj(ehr).unsqueeze(dim=1)
        # print(ehr_proj.shape)
        # 图像与EHR的注意力机制
        # context_img_ehr = self.cross_attention(q=ehr_proj, k=img_embeded, v=img_embeded)  # 1*hidden_size 不好
        context_img_ehr = self.cross_attention(q=img_embeded, k=ehr_proj, v=ehr_proj)  # 1*hidden_size #CEA

        # context_img_ehr = torch.cat((img_embeded, ehr_proj), 1)
        # # combined_embeddings = torch.cat((img_embeded, ehr_proj), 1)
        # context_img_ehr = torch.mean(context_img_ehr, dim=1, keepdim=True)

        # 分割路径
        _, hidden_states_out = self.vit(img_embeded)

        # 编码器1
        enc1 = self.encoder1(img)
        # 从ViT模型中获取的第3层隐藏状态
        x2 = hidden_states_out[3]
        # 编码器2
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        # 获取ViT的第6层隐藏状态
        x3 = hidden_states_out[6]
        # 编码器3
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        # 获取ViT的第9层隐藏状态
        x4 = hidden_states_out[9]
        # 编码器4
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))

        # 解码器4的输入
        dec4 = self.proj_feat(_, self.hidden_size, self.feat_size)
        # 解码器5
        dec3 = self.decoder5(dec4, enc4)
        # 解码器4
        dec2 = self.decoder4(dec3, enc3)
        # 解码器3
        dec1 = self.decoder3(dec2, enc2)
        # 解码器2
        dec0 = self.decoder2(dec1, enc1)
        # 输出分割结果
        seg_out = self.out(dec0)
        # 从ViT最后一层隐藏状态获取平均值
        # z12_mean = torch.mean(hidden_states_out[-1], dim=1).unsqueeze(dim=1) #z12
        z12_mean = torch.mean(hidden_states_out[9], dim=1).unsqueeze(dim=1) #z9
        # #z12_mean = torch.mean(hidden_states_out[6], dim=1).unsqueeze(dim=1) #z6
        # #z12_mean = torch.mean(hidden_states_out[3], dim=1).unsqueeze(dim=1) #z3
        
        # # 选择多个层组合        
        # layers = [3, 9, -1]  # 可自定义要组合的层
        # z_list = [torch.mean(hidden_states_out[i], dim=1).unsqueeze(dim=1) for i in layers]  # 每个z的形状: (B, 1, D)
        # z_stacked = torch.cat(z_list, dim=1)  # 拼接后形状: (B, num_layers, D)
        # z12_mean = torch.mean(z_stacked, dim=1, keepdim=True)  # 保持dim=1，形状: (B, 1, D)
        
        # print(context_img_ehr.shape)

        # 分割-注意力机制
        # context_seg_img_ehr = z12_mean  # 只有临床数据
        # context_seg_img_ehr = context_img_ehr  # 丢弃SMA

        # # 没有SMA
        # combined_embeddings = torch.cat((z12_mean, context_img_ehr), 1)
        # # combined_embeddings = torch.cat((img_embeded, ehr_proj), 1)
        # context_seg_img_ehr = torch.mean(combined_embeddings, dim=1)


        context_seg_img_ehr = self.cross_attention(q=z12_mean, k=context_img_ehr, v=context_img_ehr) #SMA


        # print(context_seg_img_ehr.shape)
        # context_seg_img_ehr = torch.mean(context_seg_img_ehr, dim=1)
        # print(z12_mean.shape, z12_mean.flatten(start_dim=1).shape, context_img_ehr.shape)

        # context_seg_img_ehr = torch.cat((z12_mean, context_img_ehr), dim=1)  # 仅临床和CT
        # context_seg_img_ehr = context_img_ehr
        # print(context_seg_img_ehr.shape)
        # 全连接层，生成分类输出
        #print(context_seg_img_ehr.flatten(start_dim=1).shape)
        # cls_output = self.fc(context_seg_img_ehr.flatten(start_dim=1))

        cls_output = self.fusion_model(context_seg_img_ehr.flatten(start_dim=1)) #new fc

        # ABLATION STUDIES
        # Clinical Only
        # ehr_proj = torch.mean(ehr_proj, dim=1)
        # cls_output = self.fusion_model(ehr_proj.flatten(start_dim=1))
        return seg_out, cls_output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from monai.networks.blocks import PatchEmbeddingBlock as PatchEmbeddingBlockOriginal
    # net = PatchEmbeddingBlock(in_channels=1, img_size=48, patch_size=16, hidden_size=48, num_heads=12)
    net = DoubleFlow(in_channels=1, out_channels=1, img_size=(48, 48, 48), feature_size=16, pos_embed='conv', patch_size=16)
    # net = UNETR(in_channels=1, out_channels=1, img_size=(48,48,48), feature_size=16, pos_embed='conv', patch_size=16)
    # net = ViT(in_channels=1, img_size=(48,48,48), patch_size=(8, 8, 8) ,pos_embed='conv')
    # net = PatchEmbeddingBlock(in_channels=1, img_size=48, patch_size=8, hidden_size=64, num_heads=4, pos_embed="conv")
    # net = CrossAttention(hidden_size=384)
    # net = KSCNet(in_channels=1, out_channels=1, img_size=(48, 48, 48), feature_size=16, patch_size=16, hidden_size=384, num_heads=12)
    # net = ehr_net()
    #net = HyMNet(in_channels=1, out_channels=1, img_size=(48, 48, 48), feature_size=16, patch_size=16, hidden_size=384, num_heads=12)

    img = torch.randn(2, 1, 48, 48, 48)
    ehr = torch.zeros(2, 15)

    # img = torch.randn(2, 216, 384)
    # ehr = torch.zeros(2, 1, 384)
    img = img.to(device)
    ehr = ehr.to(device)
    x = [img, ehr]
    net.to(device)
    seg, cls = net(img, ehr)
    # out = net(ehr)
    print(seg.shape, cls.shape)
    # seg = torch.sigmoid(seg)  # 将模型输出转换为概率值
    # seg = (seg > 0.5).float()  # 应用阈值0.5进行二值化
    # print(out.shape, torch.unique(out))