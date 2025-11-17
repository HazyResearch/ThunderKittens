"""
General helper functions for setting up experiments
"""
import os
import random

from argparse import ArgumentParser
from omegaconf import DictConfig

import torch
import numpy as np

from .logging import _format_arg


def init_wandb(args: ArgumentParser) -> any:
    """Initialize WandB"""
    if args.no_wandb:
        wandb = None
    else:
        import wandb
        wandb.init(config={},
                   entity=args.wandb_entity,
                   name=args.run_name,
                   project=args.project_name)
    return wandb


def seed_everything(seed: int) -> None:
    """
    Seed everything
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_run_name_from_checkpoint(checkpoint_path: str) -> str:
    """
    Helper function to get a condensed run name from the checkpoint path
    """
    name = []
    for s in checkpoint_path.split('/')[-1].split('-'):
        if '.pt' in s:
            name.append(f'_{s[:-3]}')
        try:
            s = s.split('=')
            s = ''.join([c[0] for c in s[1].split('_')])
            name.append(s)
        except IndexError:
            pass
    return ''.join(name)


def get_run_name_from_args(args) -> str:
    """
    Prepare a heinous identifier for the run based on args
    """
    if args.load_distill_checkpoint is not None and args.load_distill_checkpoint != 'default':
        distill_name = get_run_name_from_checkpoint(args.load_distill_checkpoint)
    else:
        distill_name = args.distill_config
    if args.load_finetune_checkpoint is not None and args.finetune_config is None:  # args.load_finetune_checkpoint != 'default':
        finetune_name = get_run_name_from_checkpoint(args.load_finetune_checkpoint)
    else:
        finetune_name = args.finetune_config
    args.run_name = f'dl-d={distill_name}-m={args.model_config}-f={finetune_name}'
    if args.no_peft_grad_ckpt is not None:
        args.run_name += f'-npgc={args.no_peft_grad_ckpt}'
    args.run_name += f'-s={args.seed}'
    if args.debug:
        args.run_name += f'-debug'
    if args.no_attention_mask is not None:
        args.run_name += f'-nam=1'
    return args.run_name.replace('True', '1').replace('False', '0')  # concise hacks


def flatten_config(config: dict, flattened: dict, key: str) -> dict:
    """
    Recursive way to flatten config args for saving to WandB
    """
    for k, v in config.items():
        if isinstance(v, dict):
            flatten_config(v, flattened, f'{key}{k}_')
        elif isinstance(v, list):
            for ix, _config in enumerate(v):
                if isinstance(_config, dict):
                    flatten_config(_config, flattened, f'{key}{k}_{ix}_')
        else:
            flattened[f'{key}{k}'] = v
    return flattened


def update_config_from_args(config: DictConfig,
                            args: ArgumentParser,
                            ignore_args: list = None) -> DictConfig:
    """
    Quick hacks to override default configs
    """
    ignore_args = [] if ignore_args is None else ignore_args
    
    # Dataset
    if getattr(args, 'dataset', None):
        config.dataset.name = args.dataset
        args.run_name += f'-ds={args.dataset}'
    
    # Optimizer
    for arg in ['lr', 'weight_decay']:
        if arg not in ignore_args:
            argval = getattr(args, arg, None)
            if argval is not None:
                setattr(config.optimizer, arg, argval)
                args.run_name += f'-{_format_arg(arg)}={argval}'
    try:
        if getattr(args, 'optim', None):
            config.optimizer.optim = args.optim
            args.run_name += f'-o={args.optim}'
    except AttributeError:
        pass
    
    # Scheduler
    try:
        if getattr(args, 'scheduler', None):
            config.lr_scheduler.lr_scheduler_type = args.scheduler
            args.run_name += f'-sc={args.scheduler}'
    except AttributeError:
        pass

    # Dataset
    for arg in [a for a in dir(args) if 'dataset_' in a]:
        argval = getattr(args, arg, None)
        if argval is not None:
            setattr(config.dataset.dataset_config, arg[len('dataset_'):], argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'

    # Dataloader
    for arg in ['batch_size']:  # , 'num_workers']:
        argval = getattr(args, arg, None)
        if argval is not None:
            setattr(config.dataloader, arg, argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'

    # Trainer
    for arg in ['gradient_accumulation_steps', 'num_train_epochs', 
                'max_steps', 'max_finetune_steps', 'eval_steps', 
                'seed', 'max_eval_batches']:
        argval = getattr(args, arg, None)
        if argval is not None:
            setattr(config.trainer, arg, argval)
            if arg in ['max_steps', 'max_finetune_steps',
                       'gradient_accumulation_steps', 'num_train_epochs', 'seed']:
                args.run_name += f'-{_format_arg(arg)}={argval}'

    # Misc
    for arg in ['replicate']:
        argval = getattr(args, arg, None)
        if argval is not None:
            args.run_name += f'-{_format_arg(arg)}={argval}'

    return config


def update_model_config_from_args(model_config: DictConfig, 
                                  args: ArgumentParser) -> DictConfig:
    """
    Override default configs given argparse args
    """
    # Overall attention 
    for arg in ['attention_type', 'learned_kernel', 'tie_qk_kernels',
                'train_qk', 'state_chunk_len', 'no_peft_grad_ckpt']:
        argval = getattr(args, arg, None)
        if argval is not None:
            setattr(model_config['attention'], arg, argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'
        else:
            try:
                getattr(model_config['attention'], arg)
            except AttributeError:
                setattr(model_config['attention'], arg, None)

    # Learned kernel
    for arg in ['lk_skip_connection', 'lk_zero_init', 'lk_normal_init']:
        argval = getattr(args, arg, None)
        if argval is not None:
            setattr(model_config['attention']['learned_kernel_kwargs'], 
                    arg[len('lk_'):], argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'
            
    # Pretrained model
    if args.pretrained_model_name_or_path is not None:  # if specified 
        pmnop = args.pretrained_model_name_or_path
        model_config.model.pretrained_model_name_or_path = pmnop
        args.run_name += f'-pmnop={pmnop.split("/")[-1]}'
        
    return model_config
