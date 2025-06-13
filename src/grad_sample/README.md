# Grad Samples with Augmentaion Multiplier

We add the augmentation multiplier based on ``GradSampleModule``. (Please see [Grad Samples in Opacus](https://github.com/pytorch/opacus/tree/main/opacus/grad_sample) for details.)

## Our mainly works
- Computes per-sample with (K) augmentation gradients: using ``opt_einsum.contract()``
- Keyword argument for augmentation multiplier: ``K: int``

In custom grad sampler methods for every trainable layer in the model, we add the ``K: int`` to support augmentation multiplier. Of course, we also need to modify the code in ``GradSampleModule`` to ``GradSampleModuleAugmented`` in ``grad_sample_module.py``.

### For example:
the modified *linear.py* 
```python
@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample_with_aug(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor, K: int
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
    if layer.weight.requires_grad:
        gs = contract("n...i,n...j->nij", backprops, activations)
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = contract("n...k->nk", backprops)
    return ret

```

### In *grad_sample_module.py* 
`line 338` add `K`:
```python
grad_samples = grad_sampler_fn(module, activations, backprops)
```
```python
grad_samples = grad_sampler_fn(module, activations, backprops, K)
```


``GradSampleModuleExpandedWeights`` is currently in early beta and can produce unexpected errors, but potentially
improves upon ``GradSampleModule`` on performance and functionality.

**TL;DR:** If you want stable implementation, use ``GradSampleModule`` (`grad_sample_mode="hooks"`).
If you want to experiment with the new functionality, you have two options. Try 
``GradSampleModuleExpandedWeights``(`grad_sample_mode="ew"`) for better performance and `grad_sample_mode=functorch` 
if your model is not supported by ``GradSampleModule``. 

Please switch back to ``GradSampleModule``(`grad_sample_mode="hooks"`) if you encounter strange errors or unexpexted behaviour.
We'd also appreciate it if you report these to us


Computes per-sample gradients for a model using backward hooks. It requires custom grad sampler methods for every
trainable layer in the model. We provide such methods for most popular PyTorch layers. Additionally, client can
provide their own grad sampler for any new unsupported layer (see [tutorial](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_grad_sampler.ipynb))

## Functorch approach
- Model wrapping class: ``opacus.grad_sample.grad_sample_module.GradSampleModule (force_functorch=True)``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="functorch"`

[functorch](https://pytorch.org/functorch/stable/) is JAX-like composable function transforms for PyTorch.
With functorch we can compute per-sample-gradients efficiently by using function transforms. With the efficient
parallelization provided by `vmap`, we can obtain per-sample gradients for any function function (i.e. any model) by 
doing essentially `vmap(grad(f(x)))`. 

Our experiments show, that `vmap` computations in most cases are as fast as manually written grad samplers used in 
hooks-based approach.

With the current implementation `GradSampleModule` will use manual grad samplers for known modules (i.e. maintain the
old behaviour for all previously supported models) and will only use functorch for unknown modules.

With `force_functorch=True` passed to the constructor `GradSampleModule` will rely exclusively on functorch. 

## ExpandedWeigths approach
- Model wrapping class: ``opacus.grad_sample.gsm_exp_weights.GradSampleModuleExpandedWeights``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="ew"`

Computes per-sample gradients for a model using core functionality available in PyTorch 1.12+. Unlike hooks-based
grad sampler, which works on a module level, ExpandedWeights work on the function level, i.e. if your layer is not
explicitly supported, but only uses known operations, ExpandedWeights will support it out of the box.

At the time of writing, the coverage for custom grad samplers between ``GradSampleModule`` and ``GradSampleModuleExpandedWeights``
is roughly the same.

## Comparative analysis

Please note that these are known limitations and we plan to improve Expanded Weights and bridge the gap in feature completeness


| xxx                          | Hooks                           | Expanded Weights | Functorch    |
|:----------------------------:|:-------------------------------:|:----------------:|:------------:| 
| Required PyTorch version     | 1.8+                            | 1.13+            | 1.12 (to be updated) |
| Development status           | Underlying mechanism deprecated | Beta             | Beta         | 
| Runtime Performance†          | baseline                       | ✅ ~25% faster  | 🟨 0-50% slower |
| Any DP-allowed†† layers       | Not supported                   | Not supported   | ✅ Supported |
| Most popular nn.* layers     | ✅ Supported                    | ✅ Supported    | ✅ Supported  | 
| torchscripted models         | Not supported                   | ✅ Supported    | Not supported |
| Client-provided grad sampler | ✅ Supported                    | Not supported   | ✅ Not needed |
| `batch_first=False`          | ✅ Supported                    | Not supported   | ✅ Supported  |
| Recurrent networks           | ✅ Supported                    | Not supported   | ✅ Supported  |
| Padding `same` in Conv       | ✅ Supported                    | Not supported   | ✅ Supported  |
| Empty poisson batches        | ✅ Supported                    | Not supported   | Not supported  |

† Note, that performance differences are unstable and can vary a lot depending on the exact model and batch size. 
Numbers above are averaged over benchmarks with small models consisting of convolutional and linear layers. 
Note, that performance differences are only observed on GPU training, CPU performance seem to be almost identical 
for all approaches.

†† Layers that produce joint computations on batch samples (e.g. BatchNorm) are not allowed under any approach    

