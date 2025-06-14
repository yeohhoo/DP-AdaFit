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
from opacus.utils.tensor_utils import sum_over_all_but_batch_and_last_n

from .utils import register_grad_sampler


@register_grad_sampler(nn.LayerNorm)
def compute_layer_norm_grad_sample_with_aug(
    layer: nn.LayerNorm,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
    K: int,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for LayerNorm

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
        K: whether or not apply augmentation multiplier
    """
    activations = activations[0]
    ret = {}
    normalize_activations = F.layer_norm(activations, layer.normalized_shape, eps=layer.eps)
    if K:
        normalize_activations = normalize_activations.reshape((-1, K,)+ (activations.shape[1:]))
        backprops = backprops.reshape((-1, K,)+ (backprops.shape[1:]))
    if layer.weight.requires_grad:
        # normalize_activations = F.layer_norm(activations, layer.normalized_shape, eps=layer.eps)
        # if K:
        #     normalize_activations = normalize_activations.reshape((-1, K,)+ (activations.shape[1:]))
        #     backprops = backprops.reshape((-1, K,)+ (backprops.shape[1:]))

        ret[layer.weight] = sum_over_all_but_batch_and_last_n(
            normalize_activations
            * backprops,
            layer.weight.dim(),
        )
    if layer.bias.requires_grad:
        ret[layer.bias] = sum_over_all_but_batch_and_last_n(backprops, layer.bias.dim())
    return ret
