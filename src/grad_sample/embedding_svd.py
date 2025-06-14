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

from typing import Dict

import torch
import torch.nn as nn
from opt_einsum import contract

from .utils import register_grad_sampler
from ..loralib.adalora import SVDEmbedding

@register_grad_sampler(SVDEmbedding)
def compute_svdembedding_grad_sample_with_aug(
    layer: SVDEmbedding, activations: torch.Tensor, backprops: torch.Tensor, K: int
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``SVDEmbedding`` layer.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    ret = {}

    if layer.weight.requires_grad or layer.r > 0:
        saved = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

        batch_size = activations.shape[0]
        if batch_size == 0:
            ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
            return ret

        index = (
            activations.unsqueeze(-1)
            .expand(*activations.shape, layer.embedding_dim)
            .reshape(batch_size, -1, layer.embedding_dim)
        )
        
        if layer.r > 0:
            # if layer.lora_A.requires_grad:
            #     lora_A_grad_sample = torch.zeros(
            #         batch_size, *layer.lora_A.shape, device=layer.lora_A.device
            #     )
            #     # print(f"lora_A_grad_sample shape={lora_A_grad_sample.shape}")

            #     # backprops_A = backprops.reshape(batch_size, -1, layer.embedding_dim)
            #     # print(f"backprops shape = {backprops.shape}")
            #     # print(f"lora_A backprops_A.shape={backprops_A.shape}")
            #     # print(f"lora_A index.shape={index.shape}")
            #     # lora_A_grad_sample.scatter_add_(
            #     #     1, index, backprops_A
            #     # )
            #     grad_sample = torch.zeros(
            #         batch_size, *layer.weight.shape, device=layer.weight.device
            #     )
            #     grad_sample.scatter_add_(
            #         1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
            #     )

            #     lora_A_grad_sample = contract("bne,er->brn",grad_sample, (layer.lora_B) * (layer.lora_E.T))
            #     # print(f"lora_A_grad_sample shape={lora_A_grad_sample.shape}")
            #     torch.backends.cudnn.deterministic = saved
            #     if K:
            #         lora_A_grad_sample = lora_A_grad_sample.reshape(
            #         (
            #             -1,
            #             K,
            #         )
            #         + (lora_A_grad_sample.shape[1:])
            #         )
            #         lora_A_grad_sample = contract("nk...->n...", lora_A_grad_sample)

            #     ret[layer.lora_A] = lora_A_grad_sample * layer.scaling / (layer.ranknum+1e-5)

            if layer.lora_E.requires_grad:
                # lora_E_grad_sample_ = torch.zeros(
                #     batch_size, *layer.lora_E.shape, device=layer.lora_E.device
                # )
                grad_sample = torch.zeros(
                    batch_size, *layer.weight.shape, device=layer.weight.device
                )
                grad_sample.scatter_add_(
                    1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
                )  #bne
                # lora_E_grad_sample = contract("bne,er->bnr", lora_E_grad_sample_, layer.lora_B)
                # "bne,er->bnr" ,nr ->br
                lora_E_grad_sample = contract("bnr,nr->br", ((grad_sample) @ (layer.lora_B)), layer.lora_A.T)
                lora_E_grad_sample = torch.unsqueeze(lora_E_grad_sample, dim=-1)

                torch.backends.cudnn.deterministic = saved
                if K:
                    lora_E_grad_sample = lora_E_grad_sample.reshape(
                    (
                        -1,
                        K,
                    )
                    + (lora_E_grad_sample.shape[1:])
                    )
                    lora_E_grad_sample = contract("nk...->n...", lora_E_grad_sample)

                ret[layer.lora_E] = lora_E_grad_sample * layer.scaling / (layer.ranknum+1e-5)

            # if layer.lora_B.requires_grad:
            #     grad_sample = torch.zeros(
            #         batch_size, *layer.weight.shape, device=layer.weight.device
            #     )
            #     grad_sample.scatter_add_(
            #         1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
            #     )
            #     torch.backends.cudnn.deterministic = saved
            #     lora_B_grad_sample = contract("bne,nr->ber", grad_sample, (layer.lora_A * layer.lora_E).T)
            #     if K:
            #         lora_B_grad_sample = lora_B_grad_sample.reshape(
            #         (
            #             -1,
            #             K,
            #         )
            #         + (lora_B_grad_sample.shape[1:])
            #         )
            #         lora_B_grad_sample = contract("nk...->n...", lora_B_grad_sample)

            #     ret[layer.lora_B] = lora_B_grad_sample * layer.scaling / (layer.ranknum+1e-5)

        if layer.weight.requires_grad:
            grad_sample = torch.zeros(
                batch_size, *layer.weight.shape, device=layer.weight.device
            )
            grad_sample.scatter_add_(
                1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
            )
            torch.backends.cudnn.deterministic = saved
            if K:
                grad_sample = grad_sample.reshape(
                (
                    -1,
                    K,
                )
                + (grad_sample.shape[1:])
                )
                grad_sample = contract("nk...->n...", grad_sample)

            ret[layer.weight] = grad_sample
        
    return ret
