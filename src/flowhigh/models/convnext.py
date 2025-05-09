from typing import Optional

import torch
from torch import nn
from torch.nn import Module
from torch.nn.utils import weight_norm


class ConvNeXtBlock(Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        hidden_dim: Optional[int] = None, # time embedding dim
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = hidden_dim is not None
        if hidden_dim:
            self.norm = AdaLayerNorm(dim, hidden_dim ,eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond is not None
            x = self.norm(x, cond)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    # def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
    def __init__(self, embedding_dim: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        # self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.scale = nn.Linear(hidden_dim, embedding_dim)
        self.shift = nn.Linear(hidden_dim, embedding_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    # def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond)
        shift = self.shift(cond)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        scale, shift = map(lambda t: t.unsqueeze(1).expand_as(x), (scale, shift))
        x = x * scale + shift
        return x


# class ResBlock1(nn.Module):
#     """
#     ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
#     but without upsampling layers.

#     Args:
#         dim (int): Number of input channels.
#         kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
#         dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
#             Defaults to (1, 3, 5).
#         lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
#             Defaults to 0.1.
#         layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
#             Defaults to None.
#     """

#     def __init__(
#         self,
#         dim: int,
#         kernel_size: int = 3,
#         dilation: Tuple[int, int, int] = (1, 3, 5),
#         lrelu_slope: float = 0.1,
#         layer_scale_init_value: Optional[float] = None,
#     ):
#         super().__init__()
#         self.lrelu_slope = lrelu_slope
#         self.convs1 = nn.ModuleList(
#             [
#                 weight_norm(
#                     nn.Conv1d(
#                         dim,
#                         dim,
#                         kernel_size,
#                         1,
#                         dilation=dilation[0],
#                         padding=self.get_padding(kernel_size, dilation[0]),
#                     )
#                 ),
#                 weight_norm(
#                     nn.Conv1d(
#                         dim,
#                         dim,
#                         kernel_size,
#                         1,
#                         dilation=dilation[1],
#                         padding=self.get_padding(kernel_size, dilation[1]),
#                     )
#                 ),
#                 weight_norm(
#                     nn.Conv1d(
#                         dim,
#                         dim,
#                         kernel_size,
#                         1,
#                         dilation=dilation[2],
#                         padding=self.get_padding(kernel_size, dilation[2]),
#                     )
#                 ),
#             ]
#         )

#         self.convs2 = nn.ModuleList(
#             [
#                 weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
#                 weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
#                 weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
#             ]
#         )

#         self.gamma = nn.ParameterList(
#             [
#                 nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
#                 if layer_scale_init_value is not None
#                 else None,
#                 nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
#                 if layer_scale_init_value is not None
#                 else None,
#                 nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
#                 if layer_scale_init_value is not None
#                 else None,
#             ]
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
#             xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
#             xt = c1(xt)
#             xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
#             xt = c2(xt)
#             if gamma is not None:
#                 xt = gamma * xt
#             x = xt + x
#         return x

#     def remove_weight_norm(self):
#         for l in self.convs1:
#             self.remove_weight_norm(l)

#         for l in self.convs2:
#             self.remove_weight_norm(l)

#     @staticmethod
#     def get_padding(kernel_size: int, dilation: int = 1) -> int:
#         return int((kernel_size * dilation - dilation) / 2)

