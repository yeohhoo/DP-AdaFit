#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

from .utils import register_grad_sampler


@register_grad_sampler(nn.GroupNorm)
def compute_group_norm_grad_sample_with_aug(
    layer: nn.GroupNorm,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
    K: int,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for GroupNorm

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
        K: whether or not apply augmentation multiplier
    """
    activations = activations[0]
    ret = {}
    normalize_activations = F.group_norm(activations, layer.num_groups, eps=layer.eps)
    if K:
        normalize_activations = normalize_activations.reshape(
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

    if layer.weight.requires_grad:
        ## gs = F.group_norm(activations, layer.num_groups, eps=layer.eps) * backprops
        # normalize_activations = F.group_norm(activations, layer.num_groups, eps=layer.eps)
        if K:
        #     normalize_activations = normalize_activations.reshape(
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

            gs = contract("nki..., nki...->ni", normalize_activations, backprops)
        else:        
            gs = contract("ni..., ni...->ni", normalize_activations, backprops)
            
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        if K:
            ret[layer.bias] = contract("nki...->ni", backprops)
        else:
            ret[layer.bias] = contract("ni...->ni", backprops)
    return ret
