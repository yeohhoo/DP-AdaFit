"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time

import numpy as np
import torch as th
import torch.distributed as dist

from opacus.validators import ModuleValidator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from guided_diffusion import dist_util_pre, logger
from guided_diffusion.nn import ExponentialMovingAverage
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    if args.model_path is None:
        raise ValueError('Need to specify a checkpoint.')
    
    dist_util_pre.setup_dist()
    sample_dir_log = os.path.join(os.path.dirname(args.model_path),"sample")
    logger.configure(dir=sample_dir_log)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.to(device)
    # model = DPDDP(model)

    # print("model state dict")
    # params = model.state_dict()
    # for k, v in params.items():
    #     print(k)
    # print("*******************************")

    state = th.load(args.model_path, map_location=device)
    # state = dist_util.load_state_dict(args.model_path, map_location=device)
    # print("load model state")
    # for k, v in state['model'].items():
    #     k = k.replace('_module.module.','')
    #     print(k)
    # print("******************************")
    # {k.replace('module.',''):v for k,v in state['model'].items()}
    # model.load_state_dict(state['model'])
    model_dict = model.state_dict()
    state = {k.replace('_module.module.',''):v for k,v in state['model'].items()}

    filter_dict = {k: v for k, v in state.items()
                               if k in model_dict}
    model.load_state_dict(filter_dict)

    if args.use_ema:
        ema = ExponentialMovingAverage(
            model.parameters(), decay=args.ema_rate)
        ema.load_state_dict(state['ema'])
        ema.copy_to(model.parameters())

    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()


        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        logger.log(f"current time {time.strftime('%x %X')}")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir_log, f"samples_{shape_str}.npz") #logger.get_dir()
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5000, # 50000
        batch_size=5000,  #6400 mnist; cifar
        use_ddim=False,
        model_path="",
        use_ema=False,  # mnist True, cifar False
        ema_rate=0.9999,
        use_ddp=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
