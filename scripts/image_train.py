"""
Train a diffusion model on images.
"""
import torch.multiprocessing as mp

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    get_module_size,

)
from guided_diffusion.dp_train_util import TrainLoop


def run_main(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    processes = []
    for rank in range(args.n_gpus_per_node):
        args.local_rank = rank 
        args.global_rank = rank + \
            args.node_rank * args.n_gpus_per_node 
        args.global_size = args.n_nodes * args.n_gpus_per_node
        print('Node rank %d, local proc %d, global proc %d' % (
            args.node_rank, args.local_rank, args.global_rank))
        p = mp.Process(target=dist_util.setup_dist, args=(args, main)) 

        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def main(args):
    # args = create_argparser().parse_args()

    # dist_util.setup_dist(args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.global_rank == 0:

        logger.configure()

        logger.log("creating model and diffusion...")


        model_size, model_size_mb, model_size_all, model_size_mb_all = get_module_size(model)
        logger.log("model argments: %s" % args)
        logger.log("model.channel_mult: %s" % (model.channel_mult,))
        logger.log('model number: {:.3f}'.format(model_size))
        logger.log('model size: {:.3f}MB'.format(model_size_mb))
        logger.log('all model number: {:.3f}'.format(model_size_all))
        logger.log('all model size: {:.3f}MB'.format(model_size_mb_all))


    model.to(args.local_rank)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    # logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        # microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        args=args  # pass for dp
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        # microbatch=-1,  # -1 disables microbatches
        ema_rate=0.9999,  # comma-separated list of EMA values
        log_interval=50,
        save_interval=50,
        snapshot_freq=0,
        resume_checkpoint="",
        resume_step=-1,
        resume_epoch=-1,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_steps=50000,
        use_pretrain=0,
    )
    privacy = dict(
        sigma=2.5,  # privacy-specify hyperparameter
        delta=1e-5,
        epsilon=10,
        poisson_sampling=True,
        max_physical_batch_size=4, #512
        max_per_sample_grad_norm=1.2,
        timestep_mul=1, # noise multiplicity

    )
    dist = dict(
        n_gpus_per_node=1,
        n_nodes=1,
        node_rank=0,
        master_address='127.0.0.1',
        master_port=65535,
        omp_n_threads=-1,
        local_rank=0,
        global_rank=0,
        global_size=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    add_dict_to_argparser(parser, privacy)
    add_dict_to_argparser(parser, dist)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    run_main(args)
