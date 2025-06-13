#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module,
                                bias: str = 'none',
                                norm: str = 'all',
                                # middle_block: str='frozen',
                                output_block: str='none',
                                input_block: str = 'none',) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        # print(f"************************cxh:{bias}")
        for n, p in model.named_parameters():
            if 'bias' in n:
                # print(f"######################cxh:{bias}")
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError
    
    if norm == 'none':
        return
    elif norm == 'all':
        for n, p in model.named_parameters():
            if 'norm' in n:
                # print(f"######################cxh:{norm}")
                p.requires_grad = True
    else:
        raise NotImplementedError
        
    if output_block == 'none':
        return
    elif output_block == 'all':
        for n, p in model.named_parameters():
            if 'output_blocks' in n:
                # print(f"output_blocks######################cxh:{n}")
                p.requires_grad = True
    else:
        raise NotImplementedError

    if input_block == 'none':
        return
    elif input_block == 'all':
        for n, p in model.named_parameters():
            if 'input_blocks' in n:
                # print(f"######################cxh:{norm}")
                p.requires_grad = True
    else:
        raise NotImplementedError



def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
