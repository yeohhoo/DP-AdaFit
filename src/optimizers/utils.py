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

from typing import List

import torch.nn as nn
from torch.optim import Optimizer


def params(optimizer: Optimizer, module: nn.Module,) -> List[nn.Parameter]:
    """
    Return all parameters controlled by the optimizer
    Args:
        optimizer: optimizer

    Returns:
        Flat list of parameters from all ``param_groups``
    """
    ret = []
    # ret_id = []
    dp_ret = []
    dp_paras = ['lora_E','weight','bias' ]  #'lora_A','lora_B',
    for n, p in module.named_parameters():
        for dp_name in dp_paras:
            if (dp_name in n) and p.requires_grad:
                # print(f"n={n}")
                dp_ret.append(id(p))

    for param_group in optimizer.param_groups:
        ret += [p for p in param_group["params"] if p.requires_grad]
        # ret_id += [id(p) for p in param_group["params"] if p.requires_grad]
    print(f"%%%%%%%%%%%%%%%%%%%%%%%lora_E={dp_ret}")
    # print(f"len(dp_ret)={len(dp_ret)}")
    # print(f"len(ret)={len(ret)}")
    # print(f"len(ret_id)={len(ret_id)}")
    # print(f"%%%%%%%%%%%%%%%%%%%%%%%optimizer={ret_id}")
    return ret, dp_ret
