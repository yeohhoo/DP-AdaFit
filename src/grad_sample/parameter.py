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
from opt_einsum import contract

from .utils import register_grad_sampler


class Parameter_linear(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(size))

    def forward(self, input):
        return self.gamma * input
        
@register_grad_sampler(Parameter_linear)
def compute_parameter_grad_sample_with_aug(
    layer: Parameter_linear, activations: List[torch.Tensor], backprops: torch.Tensor, K: int
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
        K: whether or not apply augmentation multiplier
    """
    activations = activations[0]
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
    assert activations.shape == backprops.shape
    if layer.gamma is not None and layer.gamma.requires_grad:
        # ret[layer.gamma] = contract("n...k->nk", backprops)
        # gs = contract("n...i,n...i->ni", backprops, activations)
        # ret[layer.gamma] = gs
        gs = contract("n...i,n...i->n", backprops, activations)
        ret[layer.gamma] = gs.reshape(-1,1)
        print(f"activate = {activations.shape}")
        print(f"backprops = {backprops.shape}")
        print(f"parameter.gradient.shape={ret[layer.gamma].shape}")
    return ret
