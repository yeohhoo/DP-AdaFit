import copy
import functools
import os
import random
import numpy as np
# import matplotlib.pyplot as plt
from math import ceil
import PIL
import time

from scipy import linalg
from pathlib import Path

import torch as th
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
from torchvision.utils import make_grid

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import ExponentialMovingAverage
from .resample import LossAwareSampler, UniformSampler

# from opacus import PrivacyEngine
from src.privacy_engine import PrivacyEngineAugmented
from src.utils.dataset import (get_data_loader,
                               get_data_loader_augmented,
                               populate_dataset,
                               getImagenetTransform,
                               build_transform)
from src.utils.dataloader_aug import  prepare_dataloaders

from src.loralib import (RankAllocator,
                         compute_orth_regu,
                         mark_only_lora_as_trainable,)
from src.schedulers.noise_scheduler import DynamicExponentialNoise

from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.validators import ModuleValidator

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        args,
        model,
        diffusion,
        # train_loader,
        # test_loader,
        batch_size,
        # microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.args = args
        self.model = model
        if not ModuleValidator.is_valid(self.model):
            self.model = ModuleValidator.fix(self.model)

        self.diffusion = diffusion
        # logger.log("creating data loader...")

        populate_dataset(self.args)
        ## Creating ImageNet Dataloaders with or without AugmentationMultiplicity depending on args.transform
        self.train_loader, self.test_loader = prepare_dataloaders(self.args)
        # print(type(train_loader))

        # self.train_loader = train_loader,
        # print(type(self.train_loader))
        # self.test_loader = test_loader,
        self.batch_size = batch_size
        # self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = ema_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        # self.use_fp16 = use_fp16?
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = self.args.resume_step

        # augmentation 
        self.K = self.args.transform

        self.trainable_names = ["gamma"] # ,"bias", "norm","emb_layers"

        self.loss_group = []

        if th.cuda.is_available():
            self.device = 'cuda:%d' % self.args.local_rank
            self.model = self.model.to(device=self.device)

            # use the pre_trained model
            if self.args.use_pretrain:
                if self.args.resume_checkpoint is None:
                    raise ValueError('Need to specify a checkpoint.')
                if self.args.global_rank == 0:
                    logger.log(f"loading model from checkpoint: {self.args.resume_checkpoint}...")
                state = dist_util.load_state_dict(self.args.resume_checkpoint, map_location=self.device)
                
                print(state.keys())
                print(f"self.model.state_dict().keys={self.model.state_dict().keys()}")
                model_dict = self.model.state_dict()
                filter_dict = {k: v for k, v in state.items() 
                               if k in model_dict}
                model_dict.update(filter_dict)
                
                self.model.load_state_dict(model_dict)
                # self.model.load_state_dict(state)

                # Finetune
            # self.model = finetune(self.model, self.trainable_names)
            mark_only_lora_as_trainable(self.model,bias='all')

            n, n_r, n_num, n_r_num, named_p = check_model(self.model)
            print(f"model parameter+parameter_require={n},{n_r};{n_num},{n_r_num}")
            if self.args.global_rank == 0:
                logger.log(f"dpddp_model parameter+parameter_require={n},{n_r};{n_num},{n_r_num},{named_p}")


            self.use_ddp = True
            self.dpddp_model = DPDDP(
                self.model
            )

            # use the resume checkpoint for continue training
            if self.resume_step > 0:
                # if self.args.resume_checkpoint is None:
                #     raise ValueError('Need to specify a checkpoint.')
                # if self.args.global_rank == 0:
                #     logger.log(f"loading model from checkpoint: {self.args.resume_checkpoint}...")
                # self.privacy_accountant_state_dict = self.reload_checkpoint()

                # Finetune
                # self.dpddp_model = finetune(self.dpddp_model, self.trainable_names)
                mark_only_lora_as_trainable(self.dpddp_model, bias='all')
                n, n_r, n_num, n_r_num, named_p = check_model(self.dpddp_model)
                print(f"dpddp_model parameter+parameter_require={n},{n_r};{n_num},{n_r_num}")
                if self.args.global_rank == 0:
                    logger.log(f"dpddp_model parameter+parameter_require={n},{n_r};{n_num},{n_r_num},{named_p}")

        else:
            if dist.get_world_size() > 1 and (self.args.global_rank == 0):
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.device = 'cpu'
            self.use_ddp = False
            self.dpddp_model = self.model.to(device=self.device)
            # use the pre_trained model
            if self.args.use_pretrain:
                if self.args.resume_checkpoint is None:
                    raise ValueError('Need to specify a checkpoint.')
                if self.args.global_rank == 0:
                    logger.log(f"loading model from checkpoint: {self.args.resume_checkpoint}...")
                state = dist_util.load_state_dict(self.args.resume_checkpoint, self.device)
                # need to filter dismatch state dict
                model_dict = self.model.state_dict()
                filter_dict = {k: v for k, v in state.items() 
                               if k in model_dict}
                model_dict.update(filter_dict)
                
                self.dpddp_model.load_state_dict(model_dict)
                # self.dpddp_model.load_state_dict(state)

                # Finetune
                # self.dpddp_model = finetune(self.dpddp_model, self.trainable_names)
                mark_only_lora_as_trainable(self.dpddp_model, bias='all')
                n, n_r, n_num, n_r_num, named_p = check_model(self.dpddp_model)
                print(f"dpddp_model parameter+parameter_require={n},{n_r};{n_num},{n_r_num}")
                if self.args.global_rank == 0:
                    logger.log(f"dpddp_model parameter+parameter_require={n},{n_r};{n_num},{n_r_num},{named_p}")

        self.ema = ExponentialMovingAverage(
            self.dpddp_model.parameters(), self.ema_rate
        )

        self.opt = AdamW(
            self.dpddp_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # if self.resume_step > 0:
        #     self.privacy_accountant_state_dict = self.reload_checkpoint()

        if self.args.global_rank == 0:
            self.writer = SummaryWriter(os.path.join(logger.get_dir(), 'log'))

    def wrapper_with_dp(self):
        """
        apply the PrivacyEngine 
        Returns:
            model, optimizer, train_loader
        """
        privacy_engine = PrivacyEngineAugmented()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=self.dpddp_model,
        optimizer=self.opt,
        data_loader=self.train_loader,
        target_epsilon=self.args.epsilon,
        target_delta=self.args.delta,
        epochs=self.get_epochs_from_bs(),
        # noise_multiplier=self.args.sigma,
        max_grad_norm=self.args.max_per_sample_grad_norm,
        poisson_sampling=self.args.poisson_sampling,
        K=self.args.transform
        )
        
        self.dpddp_model = model
        self.opt = optimizer
        self.train_loader = train_loader

        #  add the noise scheduler
        dns = DynamicExponentialNoise(self.opt, gamma=0.99)

        return privacy_engine, dns

    def get_epochs_from_bs(self):
        """
        output the approximate number of epochs necessary to keep our "physical constant" eta constant.
        We use a ceil, but please not that the last epoch will stop when we reach 'ref_nb_steps' steps.
        """
        assert self.args.num_steps != -1, 'step cannot be -1, please assign a value to it'

        return(ceil(self.args.num_steps*self.batch_size/len(self.train_loader.dataset)))

    def run_loop(self):
        if self.args.global_rank == 0:
            logger.log(f"there need epoch:{ceil((self.args.num_steps)*self.batch_size/len(self.train_loader.dataset))}")
            logger.log(f"there need step:{self.args.num_steps }")
            logger.log(f"the size of dataset:{len(self.train_loader.dataset)}")
        max_physical_batch_size_with_aug = \
            self.args.max_physical_batch_size \
                if self.K == 0 else self.args.max_physical_batch_size // self.K
        privacy_engine, dns = self.wrapper_with_dp()
        total_epoch = self.get_epochs_from_bs()
        if self.resume_step > 0:
            # if self.resume_step > 0:
            if self.args.resume_checkpoint is None:
                raise ValueError('Need to specify a checkpoint.')
            if self.args.global_rank == 0:
                logger.log(f"loading model from checkpoint: {self.args.resume_checkpoint}...")

            privacy_accountant_state_dict, start_epoch = self.reload_checkpoint()
            privacy_engine.accountant.load_state_dict(privacy_accountant_state_dict)
            total_epoch = total_epoch -start_epoch
            # print(self.privacy_accountant_state_dict)
        # Initialize the RankAllocator 
        if self.args.global_rank == 0:
            rank_writer = self.writer
        else:
            rank_writer = None
        rankallocator = RankAllocator(
            self.dpddp_model, lora_r=self.args.lora_r, target_rank=self.args.target_rank,
            init_warmup=self.args.init_warmup, final_warmup=self.args.final_warmup, mask_interval=self.args.mask_interval, 
            total_step=self.args.total_step, beta1=self.args.beta1, beta2=self.args.beta2, tb_writter=rank_writer,
            tb_writter_loginterval=self.log_interval, global_rank=self.args.global_rank,
        )
        for epoch in range(total_epoch):
            with BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=max_physical_batch_size_with_aug,
                optimizer=self.opt
                ) as memory_safe_data_loader:
                for batch, cond in memory_safe_data_loader:
                    # start_time = time.time()
                    # Duplicating labels for Augmentation Multiplicity.
                    # Inputs are reshaped from (N,K,*) to (N*K,*)
                    # s.t. augmentations of each image are neighbors
                    # print(f"cond:{cond}")
                    if self.K:
                        batch = batch.view([-1]+list(batch.shape[2:]))
                        cond["y"] = th.repeat_interleave(cond["y"], repeats=self.K, dim=0)
                    batch = batch.to(self.device)
                    # cond = cond.to(self.device)
                    if self.args.class_cond:
                        cond = cond
                    else:
                        cond = {}
                    cond = {
                        k: v.to(self.device)
                        for k, v in cond.items()
                    }

                    self.run_step(batch, cond, privacy_engine, epoch, dns)
                    # print(f"self.dpddp_model.named_parameters():{self.dpddp_model.named_parameters()}")
                    # if self.step % self.log_interval == 0:
                    #     logger.dumpkvs()
                    # if self.step % self.save_interval == 0:
                    #     self.save()
                    #     # Run for a finite amount of time in integration tests.
                    #     if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    #         return
                    if (not self.opt._is_last_step_skipped) and (self.step % self.log_interval == 0) and (self.args.global_rank == 0):
                        # self.log_step()
                        logger.dumpkvs()
                    if not self.opt._is_last_step_skipped :
                        grad_cosine_similarity = self.get_grad_cosine_similarity()
                        if self.step > 0 and (self.args.global_rank == 0):
                            # save gradient cosine similarity to tensorboard
                            self.writer.add_scalar("grad_cosine_similarity/train", grad_cosine_similarity, self.step)
                            logger.log('we at the epoch: %d step %d ' % (epoch + 1, self.step))
                        if (self.step * self.batch_size/len(self.train_loader.dataset) == 0) and (self.args.global_rank == 0):
                            self.get_histogram(epoch)

                        curr_rank,_ = rankallocator.update_and_mask(self.dpddp_model, self.step)
                        self.step += 1
                    
                    # if self.args.global_rank == 0:
                    #     print("befor eps calu")
                    #     print(f"self.step={self.step}")
                    #     logger.log('Eps-value after %d epochs: %.4f' %
                    #             (epoch + 1, privacy_engine.get_epsilon(self.args.delta)))
                    #     print("after eps calu")
                if self.args.global_rank == 0:
                    # print("befor eps calu")
                    logger.log('Eps-value after %d epochs: %.4f' %
                            (epoch + 1, privacy_engine.get_epsilon(self.args.delta)))
                    # print("after eps calu")
                    if privacy_engine.get_epsilon(self.args.delta) > 10:
                        self.save(epoch, privacy_engine)
                        print(f"len(self.loss_group) ={len(self.loss_group) }")
                        break

        # Save the checkpoint if self.save_interval == 0.
            if (epoch+1) % self.save_interval == 0 and self.args.global_rank == 0:
                self.save(epoch, privacy_engine)
                # n, n_r, n_num, n_r_num, named_p = check_model(self.dpddp_model)
                # logger.log(f"dpddp_model parameter+parameter_require={n},{n_r};{n_num},{n_r_num},{named_p},{len(named_p)}")

            dist.barrier()
        # Save the last checkpoint if it wasn't already saved.
        if (epoch+1) % self.save_interval !=0 and self.args.global_rank == 0:
            self.save(epoch, privacy_engine)
            # n, n_r, n_num, n_r_num, named_p = check_model(self.dpddp_model)
            # logger.log(f"dpddp_model parameter+parameter_require={n},{n_r};{n_num},{n_r_num},{named_p},{len(named_p)}")

        if self.args.global_rank == 0:
            self.writer.flush()
            self.writer.close()
        # os.system('/root/upload.sh')


    def run_step(self, batch, cond, privacy_engine, epoch, dns):
        self.forward_backward(batch, cond, epoch, dns)
        # took_step = self.mp_trainer.optimize(self.opt)
        # if took_step:
        if not self.opt._is_last_step_skipped:
            self.ema.update(self.dpddp_model.parameters())
        self._anneal_lr()
        if self.args.global_rank == 0:
            self.log_step(privacy_engine, epoch)
        dist.barrier()

    def forward_backward(self, batch, cond, epoch, dns):
        self.opt.zero_grad()
        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)

        compute_losses = functools.partial(
            self.diffusion.training_losses_with_K,
            self.dpddp_model,
            batch,
            t,
            K=self.args.timestep_mul,
            model_kwargs=cond,
            )
        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
        loss = (losses["loss"] * weights).mean()
        loss_regu = compute_orth_regu(self.dpddp_model, regu_weight=0.05)
        # loss_list = []
        # for i in range(self.args.timestep_mul):
        #     print(f"we are run{i}")
        #     losses = compute_losses()
        #     if isinstance(self.schedule_sampler, LossAwareSampler):
        #             self.schedule_sampler.update_with_local_losses(
        #                 t, losses["loss"].detach()
        #             )
        #     loss = (losses["loss"] * weights).mean()
        #     loss_list.append(loss)
        
        # loss = th.stack(loss_list).mean()
        # print(loss_list)
        # print(f"loss={loss}")
        # print(type(self.opt))

        (loss+loss_regu).backward()
        self.opt.step()

        if self.args.global_rank == 0:
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}, epoch, self.writer, self.opt, self.log_interval, self.step, loss_regu
            )

            if not self.opt._is_last_step_skipped :
                self.loss_group.append(loss.item())
                # print(f"len(self.loss_group)={len(self.loss_group)}")
                if len(self.loss_group) > 1000:  #100
                    
                    if (sum(self.loss_group[-30:-15])- sum(self.loss_group[-15:])) < 0.003:
                        dns.is_loss_decrease = -1
                        dns.is_the_early_stage = -1
                    else:
                        dns.is_loss_decrease = 0
                        dns.is_the_early_stage = -1

                dns.step()

# self.writer.add_scalar("grad_cosine_similarity/train", grad_cosine_similarity, self.step)
        dist.barrier()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self, privacy_engine, epoch):
        logger.logkv("epoch", epoch) # ceil((self.step)*self.batch_size/len(self.train_loader.dataset))
        logger.logkv("step", self.step)
        logger.logkv("timesteps augment", self.args.timestep_mul)
        # logger.logkv("samples", (self.step + self.resume_step + 1))
        # logger.logkv("epsilon", (privacy_engine.get_epsilon(self.args.delta)))
        logger.logkv("delta", (self.args.delta))
        logger.logkv("current noise multiplier",(self.opt.noise_multiplier))
        logger.logkv("current time", time.strftime('%x %X'))

    def save(self, epoch, privacy_engine):
        def save_checkpoint(state, privacy_engine):
            saved_state = {'model': state['model'].state_dict(),
                            'ema': state['ema'].state_dict(),
                            'optimizer': state['optimizer'].state_dict(),
                            'step': state['step'],
                            'epoch': state['epoch'],
                            'privacy_accountant_state_dict':privacy_engine.accountant.state_dict(),}
            logger.log(f"Saving  checkpoint at iteration {(state['step']):06d}...")
            filename = f"checkpoint_{(state['epoch']):06d}_{(state['step']):06d}.pt"
            th.save(saved_state, os.path.join(get_blob_logdir(),filename))

        state = dict(model=self.dpddp_model,
                     ema=self.ema,
                     optimizer=self.opt,
                     step=self.step,
                     epoch=epoch)
        
        save_checkpoint(state, privacy_engine)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        
        assert self.args.resume_checkpoint != "",\
              'checkpoint_path is None,please assigan a path to it'

        # checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
        if not os.path.isfile(self.args.resume_checkpoint):
            return
        if self.args.global_rank == 0:
            logger.log('Reloading checkpoint from %s ...' % self.args.resume_checkpoint)
        data = th.load(self.args.resume_checkpoint, map_location=lambda storage, loc: storage.cuda(self.args.local_rank))

        # reload model parameters
        model_filter_dict = {k.replace('_module.',''):v for k, v in data['model'].items()}
        self.dpddp_model.load_state_dict(data['model'])  # data['model']
        # self.model.load_state_dict(model_filter_dict)  # data['model']

        # ema_filter_dict = {k.replace('_module.',''):v for k, v in data['ema'].items()}
        # self.ema.load_state_dict(ema_filter_dict)  # data['ema']

        # reload optimizer
        if 'optimizer' in data and len(data['optimizer']) > 0:
            self.opt.load_state_dict(data['optimizer'])
        elif ('optimizer' in data) ^ (len(data['optimizer']) > 0):
            # warn if only one of them is available
            warnings.warn(
                f"optimizer_state_dict has {len(data['optimizer'])} items"
                f" but optimizer is {'' if self.opt else 'not'} provided."
            )

        # reload main metrics
        if 'step' in data:
            self.step = data['step']
        else:
            print("did not fimd the starting step")
        if 'epoch' in data:
            epoch = data['epoch']
        else:
            print("did not find the starting epoch")
        #self.data_loader.batch_sampler.set_step(self.step)
        if self.args.global_rank == 0:
            logger.log(f'Checkpoint reloaded. Resuming at step {self.step}')
        
        return data['privacy_accountant_state_dict'], epoch

    def get_grad_norm(self):
        total_norm = 0
        parameters = [p for p in self.opt.param_groups[0]['params'] if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def get_grad_cosine_similarity(self):
        grads_from_o = list()

        def cosine_similarity(x, y, eps=1e-08):
            # Ensure length of x and y are the same
            if len(x) != len(y) :
                return None
            
            # Compute the dot product between x and y
            dot_product = np.dot(x, y)
            
            # Compute the L2 norms (magnitudes) of x and y
            magnitude_x = np.sqrt(np.sum(x**2)) 
            magnitude_y = np.sqrt(np.sum(y**2))
            
            # Compute the cosine similarity
            cosine_similarity = dot_product / (magnitude_x * magnitude_y)  # np.max(, eps)
            
            return cosine_similarity
        
        parameters = [p for p in self.opt.param_groups[0]['params'] if p.grad is not None and p.requires_grad]
        for p in parameters:
            # param_norm = p.grad.detach().data.norm(2)
            # total_norm += param_norm.item() ** 2
            grads_from_o.append(p.grad.view(-1))
        grads_from_o = th.cat(grads_from_o)

        if self.step == 0 or self.step == self.resume_step:
            self.grads_from_o_ = grads_from_o
            return 0
        else:
            grad_cosine_similarity = cosine_similarity(self.grads_from_o_.detach().cpu().numpy(),
                                                       grads_from_o.detach().cpu().numpy())
        self.grads_from_o_ = grads_from_o

        return grad_cosine_similarity

    def get_histogram(self, epoch):
        for name, param in self.dpddp_model.named_parameters():
            if param.grad is not None and param.requires_grad:
                self.writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch)

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')

# def save_img(x, filename, figsize=None):
#     figsize = figsize if figsize is not None else (6, 6)

#     nrow = int(np.sqrt(x.shape[0]))
#     image_grid = make_grid(x, nrow)
#     plt.figure(figsize=figsize)
#     plt.axis('off')
#     plt.imshow(image_grid.permute(1, 2, 0).cpu())
#     plt.savefig(filename, pad_inches=0., bbox_inches='tight')
#     plt.close()

# def sample_random_image_batch(sampling_shape, sampler, path, device, n_classes=None, name='sample'):
#     make_dir(path)

#     x = th.randn(sampling_shape, device=device)
#     if n_classes is not None:
#         y = th.randint(n_classes, size=(
#             sampling_shape[0],), dtype=th.int32, device=device)

#     x = sampler(x, y)
#     x = x / 2. + .5

#     save_img(x, os.path.join(path, name + '.png'))
#     np.save(os.path.join(path, name), x.cpu())


# def parse_resume_step_from_filename(filename):
#     """
#     Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
#     checkpoint's number of steps.
#     """
#     split = filename.split("model")
#     if len(split) < 2:
#         return 0
#     split1 = split[-1].split(".")[0]
#     try:
#         return int(split1)
#     except ValueError:
#         return 0


def finetune(model, trainable_names):
    # Step1: Freeze all params
    for name, param in model.named_parameters():
        # param.requires_grad = False
        param.requires_grad = True

    # Step2: Unfreeze specifc params
    for name, param in model.named_parameters():
        for trainable_name in trainable_names:
            if trainable_name in name:
                # param.requires_grad = True
                param.requires_grad = False

    return model

def check_model(model):
    n = 0
    n_num = 0
    n_r = 0
    n_r_num = 0
    named_p = []
    for name, param in model.named_parameters():
        n += 1
        n_num += param.nelement()
        if param.requires_grad == True:
            n_r += 1
            n_r_num += param.nelement()
            named_p.append((name, param.shape))  #, param.grad_sample
    
    return n, n_r, n_num, n_r_num, named_p


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, epoch, writer, opt, log_interval, step, loss_regu):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        logger.logkv_mean("loss_regu", loss_regu.mean().item())
        if (not opt._is_last_step_skipped) and (step % log_interval == 0):
            writer.add_scalar("loss/train", values.mean(), epoch)

        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
