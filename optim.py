from typing import Dict, Any, Iterable
import copy

from torch.optim import *
import logging


OptimizerDict = dict(
    SGD=SGD,
    Adadelta=Adadelta,
    Adagrad=Adagrad,
    Adam=Adam,
    AdamW=AdamW,
    SparseAdam=SparseAdam,
    Adamax=Adamax,
    ASGD=ASGD,
    Rprop=Rprop,
    RMSprop=RMSprop,
    LBFGS=LBFGS
)


def get_optimizer(params: Iterable, optim_cfg: Dict[str, Any]) -> Optimizer:
    name = optim_cfg["name"]
    optimizer = OptimizerDict[name]

    kwargs = copy.deepcopy(optim_cfg)
    kwargs.pop("name")

    return optimizer(params=params, **kwargs)



""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def add_weight_decay_adaptor(model, args, weight_decay=1e-5, skip_list=()):
    logger = logging.getLogger("add weight decay")
    logger.info("Start adding weight decay")
    decay = []
    no_decay = []
    mask_real = []
    token_mask = []
    for name, param in model.named_parameters():
        if 'token_mask' in name:
            logger.info(f"token_mask {name}")
            mask_real.append(param)
        if 'adaptorFFN' in name:
            logger.info(f"adaptorFFN {name}")
            token_mask.append(param)
        if 'head' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                logger.info(f"no_decay {name}")
                no_decay.append(param)
            else:
                logger.info(f"decay {name}")
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay', 'lr': args.lr},
        {'params': decay, 'weight_decay': weight_decay, 'name': 'base_decay', 'lr': args.lr},
        {'params': mask_real, 'weight_decay': 0., 'name': 'token_mask', 'lr': args.lr_TI},
        {'params': token_mask, 'weight_decay': 0., 'name': 'adaptorFFN', 'lr': args.lr_FFN},
        ]


def create_optimizer(args, model, filter_bias_and_bn=True, backbone=True):
    logger = logging.getLogger("create_optimizer")
    logger.info("Start creating optimizer")

    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        
        if args.model == 'deit_tiny_patch16_224_MEAT' or args.model == 'deit_small_patch16_224_MEAT':
            parameters = add_weight_decay_adaptor(model, args, weight_decay, skip)
            logger.info(f"finetune model parameters name: {parameters[0]['name']}-{parameters[1]['name']}-{parameters[2]['name']}-{parameters[3]['name']}")
            logger.info(f"finetune model parameters initial weight decay: {parameters[0]['weight_decay']}-{parameters[1]['weight_decay']}-{parameters[2]['weight_decay']}-{parameters[3]['weight_decay']}")
            logger.info(f"finetune model parameters initial lr: {parameters[0]['lr']}-{parameters[1]['lr']}-{parameters[2]['lr']}-{parameters[3]['lr']}")

        else:
            parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.

    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer