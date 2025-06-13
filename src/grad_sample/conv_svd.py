import math
from typing import Dict, List, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import unfold2d, unfold3d
from opt_einsum import contract

from .utils import register_grad_sampler

from ..loralib.adalora import (SVDConv1d, SVDConv2d)


@register_grad_sampler([SVDConv1d, SVDConv2d])
def compute_svdconv_grad_sample_with_aug(
    layer: Union[SVDConv1d, SVDConv2d],
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
    K: int,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for adaptive loRA convolutional layers.
    Note: layer.group = 1
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
        K: whether or not apply augmentation multiplier
    """
    activations = activations[0]
    n = activations.shape[0]
    if n == 0:
        # Empty batch
        ret = {}
        ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.zeros_like(layer.bias).unsqueeze(0)
        return ret

    # get activations and backprops in shape depending on the Conv layer
    if type(layer) is SVDConv2d:
        # print("my code is in Conv2d") #
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    elif type(layer) is SVDConv1d:
        # print("my code is in conv1d")
        activations = activations.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        if layer.padding == "same":
            total_pad = layer.dilation[0] * (layer.kernel_size[0] - 1)
            left_pad = math.floor(total_pad / 2)
            right_pad = total_pad - left_pad
        elif layer.padding == "valid":
            left_pad, right_pad = 0, 0
        else:
            left_pad, right_pad = layer.padding[0], layer.padding[0]
        activations = F.pad(activations, (left_pad, right_pad))
        activations = torch.nn.functional.unfold(
            activations,
            kernel_size=(1, layer.kernel_size[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
    elif type(layer) is nn.Conv3d:
        # print("my code is in conv3d")
        warnings.warn(
                "Not support svd adaptive LoRA for nn.Conv3d "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )
        # activations = unfold3d(
        #     activations,
        #     kernel_size=layer.kernel_size,
        #     padding=layer.padding,
        #     stride=layer.stride,
        #     dilation=layer.dilation,
        # )
    backprops = backprops.reshape(n, -1, activations.shape[-1])

    ret = {}
    if K:
        activations = activations.reshape(
                (
                    -1,
                    K,
                )
                + (activations.shape[1:])
            )
        backprops = backprops.reshape(
                (
                    -1,
                    K,
                )
                + (backprops.shape[1:])
            )

    if layer.r > 0:
        # if layer.lora_A.requires_grad:
        #     if K:
        #         activations_A_left = layer.lora_B
        #         backprops_A = contract("nkol,or->nklr", backprops, activations_A_left)
        #         gs_ = contract("nklr,nkil->nri", backprops_A, activations)
        #         gs = gs_ * layer.lora_E
        #         n = activations.shape[0]
        #     else:
        #         activations_A_left = layer.lora_B
        #         backprops_A = contract("nol,or->nlr", backprops, activations_A_left)
        #         gs_ = contract("nlr,nil->nri", backprops_A, activations)
        #         gs = gs_ * layer.lora_E
        #     # rearrange the above tensor and extract diagonals.
        #     # gs = gs.view(
        #     #     n,
        #     #     layer.groups,
        #     #     -1,
        #     #     layer.groups,
        #     #     int(layer.in_channels / layer.groups),
        #     #     np.prod(layer.kernel_size),
        #     # )
        #     # gs = contract("ngrg...->ngr...", gs).contiguous()
            
        #     shape = [n] + list(layer.lora_A.shape)
        #     ret[layer.lora_A] = gs.view(shape) * layer.scaling / (layer.ranknum+1e-5)

        # if layer.lora_B.requires_grad:
        #     if K:
        #         activations_B = contract("ri,nkil->nkrl",(layer.lora_A * layer.lora_E), activations)
        #         gs = contract("nkol,nkrl->nor", backprops, activations_B)
        #         n = activations.shape[0]
        #     else:
        #         activations_B = contract("ri,nil->nrl",(layer.lora_A * layer.lora_E), activations)
        #         gs = contract("nol,nrl->nor", backprops, activations_B)
        #     # rearrange the above tensor and extract diagonals.
        #     # gs = gs.view(
        #     #     n,
        #     #     layer.groups,
        #     #     -1,
        #     #     layer.groups,
        #     #     int(layer.in_channels / layer.groups),
        #     #     np.prod(layer.kernel_size),
        #     # )
        #     # gs = contract("ngrg...->ngr...", gs).contiguous()
            
        #     shape = [n] + list(layer.lora_B.shape)
        #     ret[layer.lora_B] = gs.view(shape) * layer.scaling / (layer.ranknum+1e-5)

        if layer.lora_E.requires_grad:
            if K:
                activations_E = contract("nkil,ri->nklr", activations, layer.lora_A)
                backprops_E = contract("nkol,or->nklr",backprops, layer.lora_B)
                gs = contract("nklr,nklr->nr", backprops_E, activations_E)
                n = activations.shape[0]
            else:
                activations_E = contract("nil,ri->nlr", activations, layer.lora_A)
                backprops_E = contract("nol,or->nlr",backprops, layer.lora_B)
                gs = contract("nlr,nlr->nr", backprops_E, activations_E)
            
            # gs = torch.unsqueeze(gs, dim=-1)
            # ret[layer.lora_E] = gs
            # rearrange the above tensor and extract diagonals.
            # gs = gs.view(
            #     n,
            #     layer.groups,
            #     -1,
            #     layer.groups,
            #     int(layer.in_channels / layer.groups),
            #     np.prod(layer.kernel_size),
            # )
            # gs = contract("ngrg...->ngr...", gs).contiguous()
            
            shape = [n] + list(layer.lora_E.shape)
            ret[layer.lora_E] = gs.view(shape) * layer.scaling / (layer.ranknum+1e-5)


    if layer.weight.requires_grad:
        # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
        if K:
        #     activations = activations.reshape(
        #         (
        #             -1,
        #             K,
        #         )
        #         + (activations.shape[1:])
        #     )
        #     backprops = backprops.reshape(
        #         (
        #             -1,
        #             K,
        #         )
        #         + (backprops.shape[1:])
        #     )
            grad_sample = contract("nkoq,nkpq->nop", backprops, activations)
            n = activations.shape[0]
        else:
            grad_sample = contract("noq,npq->nop", backprops, activations)
        # rearrange the above tensor and extract diagonals.
        grad_sample = grad_sample.view(
            n,
            layer.groups,
            -1,
            layer.groups,
            int(layer.in_channels / layer.groups),
            np.prod(layer.kernel_size),
        )
        grad_sample = contract("ngrg...->ngr...", grad_sample).contiguous()
        shape = [n] + list(layer.weight.shape)
        ret[layer.weight] = grad_sample.view(shape)

    if layer.bias is not None and layer.bias.requires_grad:
        if K:
            # print(f"backprops.shape={backprops.shape}")
            ret[layer.bias] = contract("nkoq->no", backprops)
        else:
            ret[layer.bias] = torch.sum(backprops, dim=2)

    return ret