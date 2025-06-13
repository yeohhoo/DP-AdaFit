from typing import Dict, List

import torch
import torch.nn as nn
from opt_einsum import contract

from .utils import register_grad_sampler

from ..loralib.adalora import SVDLinear


@register_grad_sampler(SVDLinear)
def compute_svdlinear_grad_sample_with_aug(
    layer:SVDLinear, activations: List[torch.Tensor], backprops: torch.Tensor, K: int
) -> Dict[nn.Parameter, torch.Tensor]:
    """_summary_

    Args:
        layer (SVDLinear): _description_
        activations (List[torch.Tensor]): _description_
        backprops (torch.Tensor): _description_
        K (int): _description_

    Returns:
        Dict[nn.Parameter, torch.Tensor]: _description_
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
    if layer.r > 0:
        # if layer.lora_A.requires_grad:
        #     activations_A_left = layer.lora_B * (layer.lora_E.T)  # broadcast out*r
        #     backprops_A = backprops @ activations_A_left  # B*K*out, out*r -> B*K*r
        #     gs = contract("n...i,n...j->nij", backprops_A, activations)
        #     ret[layer.lora_A] = gs * layer.scaling / (layer.ranknum+1e-5)
        #     # print(f"activate = {activations.shape}")
        #     # print(f"backprops = {backprops.shape}")
        #     # print(f"svdlinear.gradient.shape = {ret[layer.lora_A].shape}")
        # if layer.lora_B.requires_grad:
        #     activations_B = activations @ (layer.lora_A * layer.lora_E).T  # B*K*in, in*r--> B*K*r
        #     gs = contract("n...i,n...j->nij", backprops, activations_B)
        #     ret[layer.lora_B] = gs * layer.scaling / (layer.ranknum+1e-5)
        if layer.lora_E.requires_grad:
            activations_E = activations @ (layer.lora_A).T  # B*K*in,in*r->B*K*r
            backprops_E = backprops @ layer.lora_B  # B*K*out,out*r ->B*K*r
            gs = contract("n...i,n...i->ni", backprops_E, activations_E)
            gs = torch.unsqueeze(gs, dim=-1)
            ret[layer.lora_E] = gs * layer.scaling / (layer.ranknum+1e-5)

    def T(w):
        return w.T if layer.fan_in_fan_out else w
    if layer.weight.requires_grad:
        gs = contract("n...i,n...j->nij", backprops, activations)
        ret[layer.weight] = T(gs)
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = contract("n...k->nk", backprops)

    return ret
