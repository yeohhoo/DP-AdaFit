import copy
import functools
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import PIL
import time

from scipy import linalg
from pathlib import Path

import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
from torchvision.utils import make_grid

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import ExponentialMovingAverage
from .resample import LossAwareSampler, UniformSampler

from opacus import PrivacyEngine
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
        data,
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
        self.data = data
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

        if th.cuda.is_available():
            self.device = 'cuda:%d' % self.args.local_rank
            self.model = self.model.to(device=self.device)

            # use the pre_trained model
            if self.args.use_pretrain:
                if self.args.resume_checkpoint is None:
                    raise ValueError('Need to specify a checkpoint.')
                logger.log(f"loading model from checkpoint: {self.args.resume_checkpoint}...")
                state = dist_util.load_state_dict(self.args.resume_checkpoint, map_location=self.device)
                self.model.load_state_dict(state)

            self.use_ddp = True
            self.dpddp_model = DPDDP(
                self.model
            )
        else:
            if dist.get_world_size() > 1:
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
                logger.log(f"loading model from checkpoint: {self.args.resume_checkpoint}...")
                state = dist_util.load_state_dict(self.args.resume_checkpoint, self.device)
                self.dpddp_model.load_state_dict(state)

        self.ema = ExponentialMovingAverage(
            self.dpddp_model.parameters(), self.ema_rate
        )

        self.opt = AdamW(
            self.dpddp_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step > 0:
            self.reload_checkpoint()

    def wrapper_with_dp(self):
        """
        apply the PrivacyEngine 
        Returns:
            model, optimizer, train_loader
        """
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=self.dpddp_model,
        optimizer=self.opt,
        data_loader=self.data,
        target_epsilon=self.args.epsilon,
        target_delta=self.args.delta,
        epochs=self.get_epochs_from_bs(),
        # noise_multiplier=self.args.sigma,
        max_grad_norm=self.args.max_per_sample_grad_norm,
        poisson_sampling=self.args.poisson_sampling,
        # K=self.args.transform
        )
        
        self.dpddp_model = model
        self.opt = optimizer
        self.data = train_loader

        return privacy_engine

    def get_epochs_from_bs(self):
        """
        output the approximate number of epochs necessary to keep our "physical constant" eta constant.
        We use a ceil, but please not that the last epoch will stop when we reach 'ref_nb_steps' steps.
        """
        assert self.args.num_steps != -1, 'step cannot be -1, please assign a value to it'

        return(ceil(self.args.num_steps*self.batch_size/len(self.data.dataset)))

    def run_loop(self):
        logger.log(f"there need epoch:{ceil((self.args.num_steps + self.resume_step)*self.batch_size/len(self.data.dataset))}")
        logger.log(f"there need step:{self.args.num_steps }")
        logger.log(f"the size of dataset:{len(self.data.dataset)}")
        privacy_engine = self.wrapper_with_dp()
        for epoch in range(self.get_epochs_from_bs()):
            with BatchMemoryManager(
                data_loader=self.data,
                max_physical_batch_size=self.args.max_physical_batch_size,
                optimizer=self.opt
                ) as memory_safe_data_loader:
                for batch, cond in memory_safe_data_loader:
                    # start_time = time.time()
                    batch = batch.to(self.device)
                    # cond = cond.to(self.device)
                    self.run_step(batch, cond, privacy_engine)
                    # if self.step % self.log_interval == 0:
                    #     logger.dumpkvs()
                    # if self.step % self.save_interval == 0:
                    #     self.save()
                    #     # Run for a finite amount of time in integration tests.
                    #     if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    #         return
                    if (not self.opt._is_last_step_skipped) and (self.step % self.log_interval == 0):
                        # self.log_step()
                        logger.dumpkvs()
                    if not self.opt._is_last_step_skipped:
                        self.step += 1

                logger.log('Eps-value after %d epochs: %.4f' %
                         (epoch + 1, privacy_engine.get_epsilon(self.args.delta)))

        # Save the checkpoint if self.save_interval == 0.
            if epoch % self.save_interval == 0 and self.args.global_rank == 0:
                self.save(epoch)
            dist.barrier()
        # Save the last checkpoint if it wasn't already saved.
        if (epoch) % self.save_interval !=0 and self.args.global_rank == 0:
            self.save(epoch)
        os.system('/root/upload.sh')


    def run_step(self, batch, cond, privacy_engine):
        self.forward_backward(batch, cond)
        # took_step = self.mp_trainer.optimize(self.opt)
        # if took_step:
        if not self.opt._is_last_step_skipped:
            self.ema.update(self.dpddp_model.parameters())
        self._anneal_lr()
        self.log_step(privacy_engine)
        dist.barrier()

    def forward_backward(self, batch, cond):
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

        loss.backward()
        self.opt.step()

        if self.args.global_rank == 0:
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

        dist.barrier()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self, privacy_engine):
        logger.logkv("epoch", ceil((self.step)*self.batch_size/len(self.data.dataset)))
        logger.logkv("step", self.step)
        logger.logkv("timesteps augment", self.args.timestep_mul)
        # logger.logkv("samples", (self.step + self.resume_step + 1))
        # logger.logkv("epsilon", (privacy_engine.get_epsilon(self.args.delta)))
        logger.logkv("delta", (self.args.delta))
        logger.logkv("current noise multiplier",(self.opt.noise_multiplier))
        logger.logkv("current time", time.strftime('%x %X'))

    def save(self, epoch):
        def save_checkpoint(state):
            saved_state = {'model': state['model'].state_dict(),
                            'ema': state['ema'].state_dict(),
                            'optimizer': state['optimizer'].state_dict(),
                            'step': state['step'],
                            'epoch': state['epoch']}
            logger.log(f"Saving  checkpoint at iteration {(state['step']):06d}...")
            filename = f"checkpoint_{(state['epoch']):06d}_{(state['step']):06d}.pt"
            th.save(saved_state, os.path.join(get_blob_logdir(),filename))

        state = dict(model=self.dpddp_model,
                     ema=self.ema,
                     optimizer=self.opt,
                     step=self.step,
                     epoch=epoch)
        
        save_checkpoint(state)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        
        assert self.resume_checkpoint != "",\
              'checkpoint_path is None,please assigan a path to it'

        # checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
        if not os.path.isfile(self.resume_checkpoint):
            return
        logger.log('Reloading checkpoint from %s ...' % self.resume_checkpointh)
        data = th.load(self.resume_checkpoint, map_location=lambda storage, loc: storage.cuda(self.args.local_rank))

        # reload model parameters
        self.dpddp_model.load_state_dict(data['model'])
        self.ema.load_state_dict(data['ema'])


        # reload main metrics
        if 'step' in data:
            self.step = data['step']
        else:
            print("did not fimd the starting step")
        #self.data_loader.batch_sampler.set_step(self.step)

        logger.log(f'Checkpoint reloaded. Resuming at step {self.step}')


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')

def save_img(x, filename, figsize=None):
    figsize = figsize if figsize is not None else (6, 6)

    nrow = int(np.sqrt(x.shape[0]))
    image_grid = make_grid(x, nrow)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).cpu())
    plt.savefig(filename, pad_inches=0., bbox_inches='tight')
    plt.close()

def sample_random_image_batch(sampling_shape, sampler, path, device, n_classes=None, name='sample'):
    make_dir(path)

    x = th.randn(sampling_shape, device=device)
    if n_classes is not None:
        y = th.randint(n_classes, size=(
            sampling_shape[0],), dtype=th.int32, device=device)

    x = sampler(x, y)
    x = x / 2. + .5

    save_img(x, os.path.join(path, name + '.png'))
    np.save(os.path.join(path, name), x.cpu())


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


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


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
