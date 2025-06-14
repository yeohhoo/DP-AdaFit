import torch
import torch.nn as nn
import grad_sample as gs
import torch.nn.functional as F

from functorch import grad, grad_and_value, vmap
from opacus.grad_sample.functorch import make_functional
from opacus.layers.dp_multihead_attention import SequenceBias
from opt_einsum import contract


def _train_step(layer, loss_fn, activations, target, grad_sample_fn, K):

    """Run a training step on model for a given batch of data

    Add the sample gradient calculate to the layer

    Parameters
    ----------
    layer : torch.nn.Module
    torch model, an instance of torch.nn.Module
    loss_fn : function
    a loss function from torch.nn.functional
    activations : input tensor  to be fed to the model
    target : label  to be fed to the model
    grad_sample_fn : the function of sample gradient for the layer
    K : augmentation multiplier
    reference link: https://github.com/suriyadeepan/torchtest/tree/master

    Return:
    -------
    a tuple: (the tensor of layer.weight.sample_grad,
              the tensor of layer.bias.sample_grad)
    For example:
        activations = torch.ones(4, 3)
        target = torch.randint(2, (4,), dtype=torch.int64)
        linear = nn.Linear(3, 2)
        loss_fn = F.cross_entropy
        grad_sample_fn = gs.compute_linear_grad_sample_with_aug
        K = 2
        _train_step(linear, loss_fn, activations, target, grad_sample_fn, K)

    """
    layer.register_forward_hook(forward_hook_wrapper('forward_hook'))

    layer.register_backward_hook(backward_hook_wrapper('backward_hook',
                                                       grad_sample_fn,
                                                       K=K))
    outputs = layer(activations)

    loss = loss_fn(outputs, target, reduction='mean') * activations.shape[0]
    loss.backward()  # auto backward

    ret = {}
    ret_sample = []
    if layer.weight.requires_grad:
        ret[layer.weight] = layer.weight.grad
        # ret_sample[linear.weight] = linear.weight.grad_sample
        ret_sample.append(layer.weight.grad_sample)
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = layer.bias.grad
        # ret_sample[linear.bias] = linear.bias.grad_sample
        ret_sample.append(layer.bias.grad_sample)

    return tuple(ret_sample)


def normal_sample_grad_for_test(model, loss_fn, batch_size, data, targets, K):
    """calculate sample gradient per batch, slower but error

    """

    def compute_grad(sample, target):
        sample = sample.unsqueeze(0)  # prepend batch dimension for processing
        target = target.unsqueeze(0)

        prediction = model(sample)
        loss = loss_fn(prediction, target)

        return torch.autograd.grad(loss, list(model.parameters()))

    def compute_sample_grads(data, targets):
        """ manually process each sample with per sample gradient """
        sample_grads = [compute_grad(data[i], targets[i])
                        for i in range(batch_size)]
        sample_grads = zip(*sample_grads)
        list_ample_grads = list(sample_grads)

        sample_grads = [torch.stack(shards) for shards in list_ample_grads]

        i = 0
        for sample_grad in sample_grads:
            if K:  # error
                sample_grad = sample_grad.reshape(
                    (
                        -1,
                        K,
                    )
                    + (sample_grad.shape[1:])
                )
                sample_grad = torch.einsum("nk...->n...", sample_grad)
            sample_grads[i] = sample_grad
            i += 1
        return sample_grads

    per_sample_grads = compute_sample_grads(data, targets)

    return per_sample_grads


def func_sample_grad(model, data, targets, loss_fn, K):
    """use functorch to calculate sample gradient,
       support the augmentation multiplier.
       the vecotrize grad function by using vmap.

    Args:
        model (_type_): _description_
        data (_type_): _description_
        targets (_type_): _description_
        loss_fn (_type_): _description_
        K (_type_): _description_

    Returns:
        tuple: (the tensor of layer.weight.sample_grad,
              the tensor of layer.bias.sample_grad)
    """
    fmodel, _fparams = make_functional(model)

    def compute_loss(params, sample, target):
        batch = sample.unsqueeze(0)
        target_ = target.unsqueeze(0)

        # predictions = functional_call(model, (params, buffers), (batch,))
        predictions = fmodel(params, batch)
        loss = loss_fn(predictions, target_)

        return loss

    ft_compute_grad = grad_and_value(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
    params = list(model.parameters())
    # print(f"params:{params}")

    ft_per_sample_grads, per_sample_losses = ft_compute_sample_grad(params,
                                                                    data,
                                                                    targets
                                                                    )

    loss = torch.mean(per_sample_losses)
    # print(f"loss:{loss}")
    if K:
        ft_per_sample_grads_by_k = []
        for ft_per_sample_grad in ft_per_sample_grads:

            ft_per_sample_grad = ft_per_sample_grad.reshape(
                (
                    -1,
                    K,
                )
                + (ft_per_sample_grad.shape[1:])
            )
            ft_per_sample_grad = torch.einsum(
                                              "nk...->n...",
                                              ft_per_sample_grad
                                             )
            ft_per_sample_grads_by_k.append(ft_per_sample_grad)

        return tuple(ft_per_sample_grads_by_k)

    return ft_per_sample_grads


def forward_hook_wrapper(name):
    def forward_hook(module, input, output):
        # print(name, ':')
        # print('input: ', input[0])
        # print('output: ', output)
        # print('\n')
        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in input])
    return forward_hook


def backward_hook_wrapper(name, grad_sample_fn, K):
    def backward_hook(module, grad_input, grad_output):
        """
        grad_input: grad of module input
        grad_output: grad of module output
        """
        # print(name, ':')
        # print('grad_input: ', grad_input)
        # print('grad_output: ', grad_output[0])
        # print('\n')
        # print(f"module.activations:{module.activations}")
        grad_sample = grad_sample_fn(module,
                                     module.activations.pop(),
                                     grad_output[0],
                                     # grad_output[0].shape[0]//K,
                                     K)

        for p, g in grad_sample.items():
            if not hasattr(p, "grad_sample"):
                p.grad_sample = None
            p.grad_sample = g

    return backward_hook


def test_conv():
    print("test conv with augmentation multiplier!")
    activations = torch.randn(4, 3, 5, 5)
    conv2 = nn.Conv2d(3, 6, 2)
    conv2_ = nn.Conv2d(3, 6, 2)
    conv2_.load_state_dict(conv2.state_dict())
    outputs = conv2(activations)

    target = torch.zeros_like(outputs)

    loss_fn = F.mse_loss
    grad_sample_fn = gs.compute_conv_grad_sample_with_aug
    K = 2
    assert_close(conv2, conv2_, loss_fn, activations,
                 target, grad_sample_fn, K)
    print("congratulations! you got the same value!!")


def test_linear():
    print("test linear with augmentation multiplier!")
    activations = torch.ones(8, 3)
    target = torch.randint(2, (8,), dtype=torch.int64)

    linear = nn.Linear(3, 2)
    linear_ = nn.Linear(3, 2)
    linear_.load_state_dict(linear.state_dict())

    loss_fn = F.cross_entropy
    grad_sample_fn = gs.compute_linear_grad_sample_with_aug
    K = 2
    per_sample_grads_n = normal_sample_grad_for_test(linear_,
                                                   loss_fn,
                                                   activations.shape[0],
                                                   activations,
                                                   target,
                                                   2)
    print(f"per_sample_grads_n:{per_sample_grads_n}")
    per_sample_grads = assert_close(linear, linear_, loss_fn, activations,
                 target, grad_sample_fn, K)
    print("congratulations! you got the same value!!")

    return per_sample_grads


def test_groupnorm():
    print("test groupnorm with augmentation multiplier!")
    activations = torch.randn(20, 6, 10, 10)
    gnorm = nn.GroupNorm(3, 6)
    gnorm_ = nn.GroupNorm(3, 6)
    gnorm_.load_state_dict(gnorm.state_dict())
    outputs = gnorm(activations)

    target = torch.zeros_like(outputs)

    loss_fn = F.mse_loss
    grad_sample_fn = gs.compute_group_norm_grad_sample_with_aug
    K = 2
    assert_close(gnorm, gnorm_, loss_fn, activations,
                 target, grad_sample_fn, K)
    print("congratulations! you got the same value!!")


def test_multihead():
    print("test multihead with augmentation multiplier!")
    activations = torch.randn(4, 20, 16)
    squencebias = SequenceBias(16, batch_first=True)
    squencebias_ = SequenceBias(16, batch_first=True)
    squencebias_.load_state_dict(squencebias.state_dict())
    outputs = squencebias(activations)

    target = torch.zeros_like(outputs)

    loss_fn = F.mse_loss

    grad_sample_fn = gs.compute_sequence_bias_grad_sample_with_aug
    K = 2
    # per_sample_grads = normal_sample_grad_for_test(linear_,
    # loss_fn, activations.shape[0], activations, target, 2)
    # print(per_sample_grads)
    assert_close(squencebias, squencebias_, loss_fn, activations,
                 target, grad_sample_fn, K)
    print("congratulations! you got the same value!!")


def test_emdedding():
    print("test emdedding with augmentation multiplier!")
    activations = torch.randint(0, 4, (4, 2))
    emb = nn.Embedding(num_embeddings=4, embedding_dim=10)
    emb_ = nn.Embedding(num_embeddings=4, embedding_dim=10)
    emb_.load_state_dict(emb.state_dict())
    outputs = emb(activations)

    target = torch.zeros_like(outputs)

    loss_fn = F.mse_loss
    grad_sample_fn = gs.compute_embedding_grad_sample_with_aug
    K = 2
    assert_close(emb, emb_, loss_fn, activations,
                 target, grad_sample_fn, K)
    print("congratulations! you got the same value!!")


def assert_close(model, model_, loss_fn, activations, target,
                 grad_sample_fn, K):
    ret_sample = _train_step(model,
                             loss_fn,
                             activations,
                             target,
                             grad_sample_fn,
                             K)

    # per_sample_grads = normal_sample_grad_for_test(linear_,
    # loss_fn, activations.shape[0], activations, target, 2)
    per_sample_grads = func_sample_grad(model_,
                                        activations,
                                        target,
                                        loss_fn,
                                        K=2)
    # print(f"per_sample_grads:{per_sample_grads}")

    for per_sample_grad, ft_per_sample_grad in zip(ret_sample,
                                                   per_sample_grads):
        assert torch.allclose(per_sample_grad,
                              ft_per_sample_grad,
                              atol=3e-3,
                              rtol=1e-5)
    
    return per_sample_grads


if __name__ == "__main__":
    # test_emdedding()
    per_sample_grads = test_linear()
    print(per_sample_grads[0].shape)
    # max_grad_norm = 1
    # per_param_norms = [
    #             g.reshape(len(g), -1).norm(2, dim=-1) for g in per_sample_grads
    #         ]
    
    # per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
    # per_sample_clip_factor = (
    #             max_grad_norm / (per_sample_norms + 1e-6)
    #         ).clamp(max=1.0)



