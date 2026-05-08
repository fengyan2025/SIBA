from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.mlp import MLPBlock
from monai.utils import optional_import
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.nets.swin_unetr import compute_mask, WindowAttention

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

__all__ = ["ViT"]


def get_norm_layer(name, features):
    if name is None:
        return nn.Identity()

    if name == 'layer':
        return nn.LayerNorm(features)

    if name == 'batch':
        return nn.BatchNorm2d(features)

    if name == 'instance':
        return nn.InstanceNorm2d(features)

    raise ValueError("Unknown Layer: '%s'" % name)


def window_partition(x, window_size):

    x_shape = x.size()
    if len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):

    if len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):


    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class ViT(nn.Module):


    def __init__(
            self,
            in_channels: int,
            img_size: Sequence[int] | int,
            patch_size: Sequence[int] | int,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            pos_embed: str = "conv",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            post_activation="Tanh",
            qkv_bias: bool = False,
            save_attn: bool = False,
            hybrid=False,
            fth=0.0,
            norm='layer'
    ) -> None:


        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_size = patch_size
        self.fth = fth
        self.img_size = img_size
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
        self.hybrid = hybrid
        if self.hybrid:
            block = SIBATransformerBlock
        else:
            block = TransformerBlock
        self.blocks = nn.ModuleList(
            [
                block(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn, norm=norm)
                for _ in range(num_layers)
            ]
        )
        self.norm = get_norm_layer(norm, hidden_size)
        if self.hybrid:

            channels = [hidden_size, 32, 64, 128]
            blocks = [
                get_conv_layer(
                    spatial_dims=2, in_channels=channels[i - 1],
                    out_channels=channels[i], conv_only=False
                )
                for i in range(1, len(channels))
            ]

            blocks.append(
                get_conv_layer(
                    spatial_dims=2, in_channels=channels[-1],
                    out_channels=1, kernel_size=1, stride=1, conv_only=False,
                    norm=None, act='sigmoid'
                )
            )
            self.binary_seg = nn.Sequential(*blocks)



    def forward(self, x, seg=None):


        x = self.patch_embedding(x)
        B, H, W = x.shape[0], self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]

        if self.hybrid:
            x_copy = x.view(B, H, W, -1).contiguous()
            x_copy = x_copy.permute(0, 3, 1, 2)
            pred_seg = self.binary_seg(x_copy)
            if seg is None:
                seg = (pred_seg > self.fth).float()

            seg = seg.squeeze(1).detach()
            seg = seg > self.fth
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            if self.hybrid:
                x = x.view(B, H, W, -1)
                x = blk(x, seg)
            else:
                x = blk(x, seg)
            hidden_states_out.append(x)
        x = self.norm(x)

        if self.hybrid:
            return x, hidden_states_out, pred_seg
        return x, hidden_states_out


class SIBATransformerBlock(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            qkv_bias: bool = False,
            save_attn: bool = False,
            window_size=(2, 2),
            norm='layer'
    ) -> None:


        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = get_norm_layer(norm, hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2 = get_norm_layer(norm, hidden_size)


        self.window_size = window_size
        no_shift = tuple(0 for _ in window_size)
        self.shift_size = tuple(i // 2 for i in window_size)
        self.blocks = nn.ModuleList(
            [
                LocalTransformerBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=no_shift if (i % 2 == 0) else self.shift_size,
                    qkv_bias=qkv_bias,
                    drop=dropout_rate,
                    norm_layer=norm
                )
                for i in range(1)
            ]
        )

        self.downsample = None
        self.bg_output = self.fg_output = None

    def global_forward(self, x, mask=None):
        b, h, w, c = x.size()

        x = x.view(b, h * w, c)
        if mask is not None:
            mask = mask.view(b, 1, -1)


        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x.view(b, h, w, c)

    def double_global_forward(self, x, mask=None):
        b, h, w, c = x.size()

        x = x.view(b, h * w, c)
        if mask is not None:
            mask = mask.view(b, 1, -1)


        out_fg = self.attn(self.norm1(x), mask)
        out_bg = self.attn(self.norm1(x), ~mask)
        out = torch.zeros(x.size()).to(x.device)
        mask = mask.squeeze(1)
        out[mask] = out_fg[mask]
        out[~mask] = out_bg[~mask]
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x.view(b, h, w, c)

    def local_forward(self, x):
        shortcut = x
        b, h, w, c = x.size()
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        wp = int(np.ceil(w / window_size[1])) * window_size[1]
        attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def forward(self, x, mask=None):
        if mask is None:
            return self.global_forward(x, mask)

        local_ = self.local_forward(x)
        global_ = self.global_forward(x, mask)
        self.bg_output = local_.detach()
        self.fg_output = global_.detach()
        x_new = torch.zeros(x.shape).to(x.device)
        x_new[mask] = global_[mask]
        x_new[~mask] = local_[~mask]
        return x_new


class LocalTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "GELU",
            norm_layer='layer',
            use_checkpoint: bool = False,
    ) -> None:


        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = get_norm_layer(norm_layer, dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x, mask_matrix):
        x = self.norm1(x)
        b, h, w, c = x.shape
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        pad_l = pad_t = 0
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape
        dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        return x


class SABlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            qkv_bias: bool = False,
            save_attn: bool = False,
    ) -> None:


        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale)
        if mask is not None:
            attn_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            att_mat = att_mat.masked_fill(~attn_mask, float('-1e9'))

        att_mat = self.softmax(att_mat)
        if self.save_attn:

            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


class SIBAUpBlock(nn.Module):


    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Sequence[int] | int,
            upsample_kernel_size: Sequence[int] | int,
            norm_name: tuple | str,
            res_block: bool = False,
            upsample: str = 'deconv'
    ) -> None:


        super().__init__()
        upsample_stride = upsample_kernel_size
        self.up = get_upsample_blk(spatial_dims=spatial_dims, in_channels=in_channels,
                                   out_channels=out_channels, upsample_stride=upsample_stride, upsample_mode=upsample)

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):

        out = self.up(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class SIBAAttUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Sequence[int] | int,
            upsample_kernel_size: Sequence[int] | int,
            norm_name: tuple | str,
            res_block: bool = False,
    ) -> None:


        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip, mask):

        out = self.transp_conv(inp)
        out = out * mask + skip * (1 - mask)

        out = self.conv_block(out)
        return out


def get_upsample_blk(spatial_dims, in_channels, out_channels, upsample_stride, upsample_mode):
    if upsample_mode == 'deconv':
        up = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_stride,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
    elif upsample_mode == 'shuffle':
        up = nn.Sequential(
            nn.PixelShuffle(upsample_stride),
            get_conv_layer(
                spatial_dims,
                in_channels // (upsample_stride ** 2),
                out_channels,
                conv_only=True
            )
        )
    elif upsample_mode == 'upconv':
        up = nn.Sequential(
            nn.Upsample(scale_factor=upsample_stride),
            get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                conv_only=True
            )
        )
    return up


class SIBAPrUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_layer: int,
            kernel_size: Sequence[int] | int,
            stride: Sequence[int] | int,
            upsample_kernel_size: Sequence[int] | int,
            norm_name: tuple | str,
            conv_block: bool = False,
            res_block: bool = False,
            upsample: str = 'deconv'
    ) -> None:


        super().__init__()

        upsample_stride = upsample_kernel_size
        self.up = get_upsample_blk(spatial_dims=spatial_dims, in_channels=in_channels,
                                   out_channels=out_channels, upsample_stride=upsample_stride, upsample_mode=upsample)

        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_upsample_blk(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                upsample_stride=upsample_stride,
                                upsample_mode=upsample),
                            UnetResBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_upsample_blk(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                upsample_stride=upsample_stride,
                                upsample_mode=upsample),
                            UnetBasicBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_upsample_blk(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        upsample_stride=upsample_stride,
                        upsample_mode=upsample)
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.up(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class SIBABasicBlock(nn.Module):


    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Sequence[int] | int,
            stride: Sequence[int] | int,
            norm_name: tuple | str,
            res_block: bool = False,
    ) -> None:


        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        return self.layer(inp)


class UnetOutBlock(nn.Module):
    def __init__(
            self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=1,
            dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            bias=True,
            act="Tanh",
            norm=None,
            conv_only=False,
        )

    def forward(self, inp):
        return self.conv(inp)
