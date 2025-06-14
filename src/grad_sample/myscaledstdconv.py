# # use for NFResNet

# '''
# Adapted from timm
# '''

# #!/usr/bin/env python3
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# from typing import Dict, Union

# import numpy as np
# import torch
# import torch.nn as nn

# from .utils import register_grad_sampler
# from .conv import compute_conv_grad_sample_with_aug


# import torch.nn.functional as F
# # import sys
# # sys.path.append("")
# from tan4test.model_class import (MyScaledStdConv2d, 
#                                      MyScaledStdConv2dSame,
#                                      Expand,
#                                      unsqueeze_and_copy,
#                                      get_standardized_weight)


# import math
# from typing import Dict, List, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from opacus.utils.tensor_utils import unfold2d, unfold3d
# from opt_einsum import contract

# from .utils import register_grad_sampler


# def compute_stdconv_grad_sample_with_aug(
#     layer: nn.Conv2d,
#     activations: List[torch.Tensor],
#     backprops: torch.Tensor,
#     K: int,
# ) -> Dict[nn.Parameter, torch.Tensor]:
#     """
#     Computes per sample gradients for convolutional layers.

#     Args:
#         layer: Layer
#         activations: Activations
#         backprops: Backpropagations
#         K: whether or not apply augmentation multiplier
#     """
#     activations = activations[0]
#     n = activations.shape[0]
#     if n == 0:
#         # Empty batch
#         ret = {}
#         ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
#         if layer.bias is not None and layer.bias.requires_grad:
#             ret[layer.bias] = torch.zeros_like(layer.bias).unsqueeze(0)
#         return ret

#     # get activations and backprops in shape depending on the Conv layer
#     activations = unfold2d(
#                 activations,
#                 kernel_size=layer.kernel_size,
#                 padding=layer.padding,
#                 stride=layer.stride,
#                 dilation=layer.dilation,
#             )

#     backprops = backprops.reshape(n, -1, activations.shape[-1])

#     ret = {}
#     if layer.weight.requires_grad:
#         # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
#         if K:
#             activations = activations.reshape(
#                 (
#                     -1,
#                     K,
#                 )
#                 + (activations.shape[1:])
#             )
#             backprops = backprops.reshape(
#                 (
#                     -1,
#                     K,
#                 )
#                 + (backprops.shape[1:])
#             )
#             grad_sample = contract("nkoq,nkpq->nop", backprops, activations)
#             n = activations.shape[0]
#         else:
#             grad_sample = contract("noq,npq->nop", backprops, activations)
#         # rearrange the above tensor and extract diagonals.
#         grad_sample = grad_sample.view(
#             n,
#             layer.groups,
#             -1,
#             layer.groups,
#             int(layer.in_channels / layer.groups),
#             np.prod(layer.kernel_size),
#         )
#         grad_sample = contract("ngrg...->ngr...", grad_sample).contiguous()
#         shape = [n] + list(layer.weight.shape)
#         ret[layer.weight] = grad_sample.view(shape)

#     if layer.bias is not None and layer.bias.requires_grad:
#         if K:
#             ret[layer.bias] = contract("nkoq->no", backprops)
#         else:
#             ret[layer.bias] = torch.sum(backprops, dim=2)

#     return ret


# @register_grad_sampler([MyScaledStdConv2d])
# def compute_wsconv_grad_sample_with_aug(layer: MyScaledStdConv2d,activations: torch.Tensor,backprops: torch.Tensor,K: int) -> Dict[nn.Parameter, torch.Tensor]:
#     # print("type(layer):",type(layer))
#     ret = compute_stdconv_grad_sample_with_aug(layer, activations, backprops, K)
#     activations = activations[0]
#     if K:
#         batch_size = activations.shape[0] // K
#     else:
#         batch_size = activations.shape[0]

#     with torch.enable_grad():
#         weight_expanded = unsqueeze_and_copy(layer.weight, batch_size)
#         gain_expanded = unsqueeze_and_copy(layer.gain, batch_size)

#         std_weight = get_standardized_weight(weight = weight_expanded,gain=gain_expanded,eps=layer.eps)
#         std_weight.backward(ret[layer.weight])
#     ret[layer.weight] = weight_expanded.grad.clone() #erased copy?
#     ret[layer.gain] = gain_expanded.grad.clone()

#     return ret

# @register_grad_sampler([MyScaledStdConv2dSame])
# def compute_wsconv_grad_sample_with_aug(layer: MyScaledStdConv2dSame,activations: torch.Tensor,backprops: torch.Tensor, K: int) -> Dict[nn.Parameter, torch.Tensor]:
#     ret = compute_stdconv_grad_sample_with_aug(layer, activations, backprops, K)
#     if K:
#         batch_size = activations.shape[0] // K
#     else:
#         batch_size = activations.shape[0]

#     with torch.enable_grad():
#         weight_expanded = unsqueeze_and_copy(layer.weight, batch_size)
#         gain_expanded = unsqueeze_and_copy(layer.gain, batch_size)

#         std_weight = get_standardized_weight(weight = weight_expanded,gain=gain_expanded,eps=layer.eps)
#         std_weight.backward(ret[layer.weight])
#     ret[layer.weight] = weight_expanded.grad.clone() #erased copy?
#     ret[layer.gain] = gain_expanded.grad.clone()

#     return ret


# @register_grad_sampler([Expand])
# def compute_expand_grad_sample_with_aug(
#     layer,
#     activations,
#     backprops,
#     K
# ):
#     """
#     Computes per sample gradients for expand layers.
#     """
#     if K:
#         return {layer.weight: backprops.reshape((-1,K)+(backprops.shape[1:])).sum(1)}

#     return {layer.weight: backprops}
