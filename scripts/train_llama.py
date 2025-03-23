import os
import sys
from typing import Optional, cast
# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import functools
import logging
import os
import re
import warnings
from collections import OrderedDict
from typing import Any, ContextManager, Iterable, Optional, Union

import torch
from composer.core import Algorithm, Callback, Evaluator
from composer.loggers import LoggerDestination
from composer.models import ComposerModel
from composer.optim.scheduler import ComposerScheduler
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.distributed.checkpoint import LoadPlanner, SavePlanner
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry import registry
from llmfoundry.callbacks import EvalGauntlet
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.eval.datasets.in_context_learning_evaluation import (
    get_icl_task_dataloader,
)
from llmfoundry.utils.config_utils import to_dict_container, to_list_container
from llmfoundry.utils.registry_utils import construct_from_registry
from llmfoundry.utils.warnings import experimental_function

log = logging.getLogger(__name__)
# nohup: ignoring input
# Traceback (most recent call last):
#   File "/home/jupyter/PS/new3/llm-foundry/scripts/train/train_llama.py", line 25, in <module>
#     from train.mllama import Llama
# ModuleNotFoundError: No module named 'train.mllama'; 'train' is not a package

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# import src.hf_bert as hf_bert_module
# import src.create_bert as bert_module
# import src.text_data as text_data_module
# from src.optim.create_param_groups import create_param_groups
from composer import Trainer, algorithms
from composer.callbacks import (LRMonitor, MemoryMonitor, OptimizerMonitor, RuntimeEstimator, SpeedMonitor)
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler, CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
# from mllama import Llama
from mllama import Llama


def get_config(
        conf_path: str = '/home/jupyter/PS/new3/llm-foundry/scripts/train/yamls/pretrain/gpt-neo-125m.yaml',
) -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    return cast(DictConfig, test_cfg)


om_cfg = get_config(conf_path='/home/jupyter/PS/new3/llm-foundry/scripts/train/yamls/pretrain/gpt-neo-125m.yaml')



def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} '
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            f'to be divisible by world size, {dist.get_world_size()}.')
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f'WARNING: device_train_microbatch_size > device_train_batch_size, '
                f'will be reduced from {device_microbatch_size} -> {device_train_batch_size}.'
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


def build_algorithm(name, kwargs):
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    elif name == 'alibi':
        return algorithms.Alibi(**kwargs)
    elif name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == 'gated_linear_units':
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == 'low_precision_layernorm':
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1),
                            gpu_flops_available=kwargs.get(
                                'gpu_flops_available', None))
    elif name == 'runtime_estimator':
        return RuntimeEstimator()
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get(
            'log_optimizer_metrics', True), )
    elif name == 'health_checker':
        return HealthChecker(**kwargs)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                                  alpha_f=cfg.alpha_f)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                         alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def create_param_groups(
        model: torch.nn.Module,
        optimizer_config: Optional[dict[str, Any]] = None,
) -> Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]:
    """Extracts parameter groups defined in the optimizer config.

    The optimizer_config defines the optimizer args. It can additionally have key
    `disable_grad` which is a string or list of strings. If a string matches a
    parameter name, then that parameter will have `requires_grad=False`. This is
    useful for freezing parameters. It can additionally have a key
    `param_groups` which is a list of dicts. In this dict, key `param_str_match`
    defines a string; if a parameter name contains this string, then it will be
    in this parameter group. This is useful for grouping parameters together.
    The dict can also contain any other key that is a valid optimizer arg.
    Note: to handle name overlap conflicts, params are assigned to parameter
    groups and added to `param_groups` in the order that `param_str_match` appear
    in `param_groups`.

    Usage
    To disable gradient for all parameters that contain the string "norm" or "bias":
    ```
    optimizer_config: {
        "name": "decoupled_lionw",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "disable_grad": ["norm", "bias"]
    }
    ```

    To create and modify the optimizer parameters for all parameters that contain
    the string "norm" and "bias" separately:
    ```
    optimizer_config: {
        "name": "decoupled_lionw",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "param_groups": [
            {
                "param_str_match": "norm",
                "lr": 1e-4,
                "weight_decay": 0.0,
            },
            {
                "param_str_match": "bias",
                "lr": 5e-4,
                "weight_decay": 0.0,
            },
        ],
    }
    ```

    Args:
        model (torch.nn.Module): model to extract parameters from
        optimizer_config (Dict[str, Any]): optimizer config

    Returns:
        Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]: an iterable of
            torch.Tensor's or dict's. Specifies what Tensors should be optimized
            and their param groupings.
    """
    if optimizer_config is None:
        return model.parameters()

    if 'disable_grad' in optimizer_config.keys():
        str_matches = optimizer_config.pop('disable_grad')
        if isinstance(str_matches, str):
            str_matches = [str_matches]
        for str_match in str_matches:
            for n, p in model.named_parameters():
                if re.search(str_match, n):
                    p.requires_grad = False
                    log.debug(f'Setting `{n}.requires_grad = False`.')

    param_groups_config = optimizer_config.pop('param_groups', None)
    if param_groups_config is not None:
        params = []
        param_dict = OrderedDict((n, p) for n, p in model.named_parameters())

        log.debug(f'Default optimizer settings: {optimizer_config}.')
        for param_group_config in param_groups_config:
            str_match = param_group_config.pop('param_str_match')
            filter_fn = functools.partial(re.search, str_match)
            param_names = [n for n in param_dict.keys() if filter_fn(n)]
            group_params = {'params': [param_dict.pop(n) for n in param_names]}
            group_params.update(param_group_config)

            log.debug(
                f'Creating optimizer param_group with parameters: {param_names} ' + \
                f'(extracted using {str_match=}). The param_group optimizer ' + \
                f'setting overrides are: {param_group_config}.')

            params.append(group_params)

        params.insert(0, {'params': param_dict.values()})
        return params

    return model.parameters()


def build_optimizer(cfg, model):
    # if cfg.name == 'decoupled_adamw':
    return DecoupledAdamW(create_param_groups(om_cfg, om_cfg['model']),
                          lr=om_cfg['optimizer']['lr'],
                          betas=om_cfg['optimizer']['betas'],
                          eps=om_cfg['optimizer']['eps'],
                          weight_decay=om_cfg['optimizer']['weight_decay'])

    # elif cfg.name == 'adamw':
    #     from torch.optim import AdamW
    #     return AdamW(create_param_groups(None, model),
    #                 lr=cfg.lr,
    #                 betas=cfg.betas,
    #                 eps=cfg.eps,
    #                 weight_decay=cfg.weight_decay)

    # else:
    #     raise ValueError(f'Not sure how to build optimizer: {cfg.name}')


def build_dataloader(cfg, tokenizer, device_batch_size):
    if cfg.name == 'text':
        return text_data_module.build_text_dataloader(cfg, tokenizer,
                                                      device_batch_size)
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')


MASTER_CONFIG = {
    'n_layers': 24,  # 16,
    'epochs': 10,
    'log_interval': 10,
    'vocab_size': 50368,  # len(vocab),
    'd_model': 2048,
    'context_window': 16,  # 256,
    'n_heads': 16,
    'batch_size': 2,
}


def build_model(cfg: DictConfig):
    # if cfg.name == 'hf_bert':
    #     return hf_bert_module.create_hf_bert_mlm(
    #         pretrained_model_name=cfg.pretrained_model_name,
    #         use_pretrained=cfg.get('use_pretrained', None),
    #         model_config=cfg.get('model_config', None),
    #         tokenizer_name=cfg.get('tokenizer_name', None),
    #         gradient_checkpointing=cfg.get('gradient_checkpointing', None))
    # elif cfg.name == 'bert':
    #     return bert_module.create_bert_mlm(
    #         pretrained_model_name=cfg.pretrained_model_name,
    #         pretrained_checkpoint=cfg.get('pretrained_checkpoint', None),
    #         model_config=cfg.get('model_config', None),
    #         tokenizer_name=cfg.get('tokenizer_name', None),
    #         gradient_checkpointing=cfg.get('gradient_checkpointing', None))
    # if cfg.name == 'Llama':

    return Llama(om_cfg,
                 tokenizer_name=om_cfg['tokenizer']['name'])
                 # gradient_checkpointing=om_cfg['gradient_checkpointing'])
    # else:


#     raise ValueError(f'Not sure how to build model with name={cfg.name}')


# def build_model(cfg: DictConfig):
#     if cfg.name == 'llama':
#         return Llama(
#             model_config=cfg.get('model_config', {}),  # Handle potential missing config
#             tokenizer_name=cfg.get('tokenizer_name', None),
#             gradient_checkpointing=cfg.get('gradient_checkpointing', None)
#         )
#     elif cfg.name == 'mpt_causal_lm':
#         from scripts.train.mpt_causal_lm import MPTCausalLM
#         # Extract parameters from the config.  This is crucial!
#         model_params = {
#             'd_model': cfg.d_model,
#             'n_heads': cfg.n_heads,
#             'n_layers': cfg.n_layers,
#             'expansion_ratio': cfg.expansion_ratio,
#             'max_seq_len': cfg.max_seq_len,
#             'vocab_size': cfg.vocab_size,
#             'attn_impl': cfg.attn_config.attn_impl,
#             'loss_fn': cfg.loss_fn,
#             'init_device': cfg.init_device,
#             # Add any other necessary parameters here
#         }
#         return MPTCausalLM(**model_params) # Use ** to unpack the dictionary
#     else:
#         raise ValueError(f'Not sure how to build model with name={cfg.name}')


# from omegaconf import DictConfig

# def build_model(cfg: DictConfig):
#     if cfg.name == 'llama':
#         return Llama(
#             model_config=cfg.get('model_config', None),
#             tokenizer_name=cfg.get('tokenizer_name', None),
#             gradient_checkpointing=cfg.get('gradient_checkpointing', None)
#         )
#     elif cfg.name == 'mpt_causal_lm':
#         from scripts.train.mpt_causal_lm import MPTCausalLM  # Replace with the actual import
#         return MPTCausalLM(
#             d_model=cfg.d_model,
#             n_heads=cfg.n_heads,
#             n_layers=cfg.n_layers,
#             expansion_ratio=cfg.expansion_ratio,
#             max_seq_len=cfg.max_seq_len,
#             vocab_size=cfg.vocab_size,
#             attn_impl=cfg.attn_config.attn_impl,
#             loss_fn=cfg.loss_fn,
#             init_device=cfg.init_device
#         )
#     else:
#         raise ValueError(f'Not sure how to build model with name={cfg.name}')


def run(cfg: DictConfig,
        return_trainer: bool = False,
        do_train: bool = True) -> Optional[Trainer]:
    print('Training using config: ')
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print('Initializing model...')
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

    # Dataloaders
    print('Building train loader...')
    train_loader = build_dataloader(
        om_cfg['train_loader'],
        om_cfg['tokenizer'],
        om_cfg['global_train_batch_size'] // dist.get_world_size(),
    )
    print('Building eval loader...')
    global_eval_batch_size = om_cfg['global_train_batch_size']
    eval_loader = build_dataloader(
        om_cfg['eval_loader'],
        om_cfg['tokenizer'],
        global_eval_batch_size // dist.get_world_size(),
    )

    # Optimizer
    optimizer = build_optimizer(om_cfg['optimizer'], model)

    # Scheduler
    scheduler = build_scheduler(om_cfg['scheduler'])

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (om_cfg['loggers']).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in (om.cfg['callbacks']).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (om_cfg['algorithms']).items()
    ]

    if om_cfg['run_name'] is None:
        om_cfg['run_name'] = os.environ.get('COMPOSER_RUN_NAME', 'bert')

    # Build the Trainer
    trainer = Trainer(
        run_name=om_cfg['run_name'],
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=om_cfg(['eval_subset_num_batches']),
        eval_subset_num_batches=om_cfg(['eval_subset_num_batches']),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get('device', None),
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
        python_log_level=cfg.get('python_log_level', None),
        autoresume=cfg.get('autoresume'),
    )

    print('Logging config...')
    log_config(om_cfg)

    if do_train:
        print('Starting training...')
        trainer.fit()

    if return_trainer:
        return trainer


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    run(cfg)