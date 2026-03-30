# The encoder h consists of the rst two blocks of an ImageNet
#  pre-trained ResNet18 followed by a projector, where the projector is composed of a 1x1
#  convolutional layer and an adaptive pooling layer
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from torch import nn
import torch
from src.models.networks.resnet import generate_model


class SegMapEncoder(nn.Module):
    def __init__(self):
        super(SegMapEncoder, self).__init__()
        self.resnet = generate_model(18)
        # self.b1 = self.resnet.layer1
        # self.b2 = self.resnet.layer2
        self.conv1 = nn.Conv3d(in_channels=128, out_channels=49, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool3d((12, 8, 8))
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, 768))
        self.indicator_token = nn.Parameter(torch.randn(1, 768))
        self.extra_token = nn.Parameter(torch.randn(2, 768))
    def forward(self, x):
        current_batch_size = x.size(0)
        x = self.resnet(x)[2]
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        # x = x.unsqueeze(dim=1)
        x = x + self.position_embedding + self.indicator_token.unsqueeze(0)
        # x = torch.cat([self.indicator_token.unsqueeze(0), x], dim=1)
        # print(x.shape, self.extra_token.unsqueeze(0).repeat(current_batch_size, 1, 1).shape)
        x = torch.cat([x, self.extra_token.unsqueeze(0).repeat(current_batch_size, 1, 1)], dim=1)

        # print(x.shape)
        return x



# from monai.networks.nets.vit import ViT
from monai.networks.blocks.transformerblock import TransformerBlock
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Sequence, Tuple, Union
from monai.utils import ensure_tuple_rep
from monai.networks.blocks import PatchEmbeddingBlock

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
class SegPromptBackbone(nn.Module):

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
        # frozen
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

        self.img_embed = PatchEmbeddingBlock(
            in_channels=in_channels, img_size=img_size, patch_size=patch_size, hidden_size=hidden_size,
            num_heads=num_heads)
        self.cls_token = nn.Parameter(torch.zeros(1, hidden_size))


        # seg map encoder
        self.seg_map_encoder = SegMapEncoder()
        # linear classifer
        self.classifier = nn.Linear(79*768, 1)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    def forward(self, img, mask):
        current_batch_size = img.size(0)
        img_embed = self.img_embed(img)
        cls_token = self.cls_token
        cls_token = cls_token.unsqueeze(0).repeat(current_batch_size, 1, 1)
        seg_encoder = self.seg_map_encoder(mask)
        # print(cls_token.shape, img_embed.shape, seg_encoder.shape)
        z = torch.cat((cls_token, img_embed, seg_encoder), dim=1)
        # print(z.shape)
        _, hidden_states_out = self.vit(z)

        logits = self.classifier(hidden_states_out[-1].flatten(start_dim=1))
        # print(logits.shape, logits)
        # print(hidden_states_out[-1].shape)
        return logits


if __name__ == '__main__':
    model = SegPromptBackbone(in_channels=1, out_channels=1, img_size=48)
    # model = SegMapEncoder()
    img = torch.randn(2, 1, 48, 48, 48)
    mask = torch.randn(2, 1, 48, 48, 48)
    out = model(img, mask)
    print(out.shape)