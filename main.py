import argparse
import datetime
from re import A
import numpy as np
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import logging
import os
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, MaskKeepLoss
from samplers import RASampler
import utils
import models


def get_args_parser():
    parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--distill', type=bool, default=False)
    parser.add_argument('--distillw', type=float, default=0.5, help='distill rate (default: 0.5)')

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch16_224_MEAT', type=str, metavar='MODEL',
                        help='Name of model to train') 
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--save_ep_freq', default=10, type=int, help='save epoch frequency')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_TI', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_FFN', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 5e-4)')

    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.0)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher_model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher_path', type=str, default='')
    parser.add_argument('--distillation_type', default='none',
                        choices=['none', 'soft', 'hard', 'cnn_soft', 'cnn_hard', 'sd', 'pos', 'pos_sd'],
                        type=str, help="")
    parser.add_argument('--distillation_alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation_tau', default=1.0, type=float, help="") 
    parser.add_argument('--pos_alpha', default=0.01, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default='dataset/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='CIFAR100',
    choices=['CIFAR100', 'CUB2011', 'Car', 'Dogs', 'Flowers', 
        'FGVC', 'WikiArt', 'Sketches', 'Places365',
        'Seq1', 'Seq2', 'Seq3',
        'IMNET', 'INAT', 'INAT19', 'CIFAR32', 'CIFAR100_CNN'],
    type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def create_model(args):
    logger = logging.getLogger("create_model")
    if args.model == 'deit_tiny_patch16_224_MEAT':
        model = models.deit_tiny_patch16_224_MEAT(
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path
        )
    elif args.model == 'deit_small_patch16_224_MEAT':
        model = models.deit_small_patch16_224_MEAT(
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path
        )
    return model


def get_dataloader(args, dataset_train, dataset_val):
    logger = logging.getLogger("get_dataloader")
    logger.info("get_dataloader")
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    return data_loader_train, data_loader_val


def get_data_seq(args, dataset_seq):
    logger = logging.getLogger("Get dataset sequence")
    logger.info("Get dataset dequence")

    data = {}
    taskcla = []

    n = 0
    for idx, t in enumerate(dataset_seq):
        data[idx] = {}
        args.data_set = t
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(is_train=False, args=args)
        logger.info("Dataset {} num_classes: {}".format(args.data_set, args.nb_classes))
        logger.info("train {} test: {}".format(len(dataset_train), len(dataset_val)))
        
        data_loader_train, data_loader_val = get_dataloader(
                                args=args,
                                dataset_train=dataset_train,
                                dataset_val=dataset_val)
        
        data[idx]['name'] = t
        data[idx]['train'] = data_loader_train
        data[idx]['val'] = data_loader_val
        data[idx]['ncla'] = args.nb_classes
        taskcla.append((idx, data[idx]['ncla']))
        n += args.nb_classes
    data['ncla'] = n

    return data, taskcla


def main(args):

    utils.init_distributed_mode(args)
    
    if args.eval: # evaluation only
        logfile_dir = os.path.join(args.output_dir, "eval-logs")
    else: # training
        logfile_dir = os.path.join(args.output_dir, "train-logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    tb_dir = os.path.join(args.output_dir, "tf-logs")
    tb_log_dir = os.path.join(tb_dir, args.model+ "_" + args.data_set)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(
            tb_dir,
            args.model+ "_" + args.data_set
        ),
        flush_secs=1
    )
    logger = utils.get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            logfile_dir,
            args.model+ "_" + args.data_set + ".log"
        )
    )

    logger.info("Start running with args: \n{}".format(args))
    logger.info("Distributed: {}".format(args.distributed))
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if args.data_set == 'Seq1':
        dataset_seq = ['IMNET12', 'CIFAR100', 'CUB2011', 'FGVC', 'Sketches', 
                    'WikiArt', 'Car']
    logger.info(f"dataset sequence: {dataset_seq}")

    dataloaders, taskcla = get_data_seq(args=args, 
            dataset_seq=dataset_seq)
    args.taskcla = taskcla
    logger.info(f'Task info = {taskcla}')


    # get dataloaders
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    logger.info("Dataset {} num_classes: {}".format(args.data_set, args.nb_classes))
    logger.info("train {} test: {}".format(len(dataset_train), len(dataset_val)))

    # if True:  # args.distributed:
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn_list = []
    for t, ncla in taskcla:
        nb_classes = ncla
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=nb_classes) # num_classes=args.nb_classes
        mixup_fn_list.append(mixup_fn)
    logger.info(f"Creating model: {args.model}")
    model = create_model(args=args)

    logger.info("Student model: {}".format(model))

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        depth = 12
        for i in range(depth):
            print(f"depth {i}")
            model.blocks[i].mlp.fc1.weight.data.copy_(checkpoint_model["blocks.{}.mlp.fc1.weight".format(str(i))])
            model.blocks[i].mlp.fc1.bias.data.copy_(checkpoint_model["blocks.{}.mlp.fc1.bias".format(str(i))])
            model.blocks[i].mlp.fc2.weight.data.copy_(checkpoint_model["blocks.{}.mlp.fc2.weight".format(str(i))])
            model.blocks[i].mlp.fc2.bias.data.copy_(checkpoint_model["blocks.{}.mlp.fc2.bias".format(str(i))])
            logger.info(f"block {i} fc1, fc2 weights and bias loaded.")
        else:
            model.load_state_dict(checkpoint_model, strict=False)
        logger.info("Pretrained weights loaded!")

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        logger.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()
    
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = MaskKeepLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    logger.info("Criterion: {}".format(criterion))

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    logger.info(f"Start training for {args.epochs} epochs")
    for t, tncla in taskcla[1:]:
        data_loader_train = dataloaders[t]['train']
        data_loader_val = dataloaders[t]['val']

        logger.info(f"Start training {dataloaders[t]['name']} for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                logger.info("distributed, data_loader_train set epoch")
                data_loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model=model, model_name=args.model,
                criterion=criterion, data_loader=data_loader_train,
                optimizer=optimizer, device=device, epoch=epoch, loss_scaler=loss_scaler,
                max_norm=args.clip_grad, model_ema=model_ema, mixup_fn_list=mixup_fn_list,
                tb_writer=tb_writer, iteration=__global_values__["it"], 
                distillation_type=args.distillation_type, args=args, 
                task=t,
                set_training_mode=True #set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            )
            logger.info("Averaged stats:")
            logger.info(train_stats)
            __global_values__["it"] += len(data_loader_train)
            tb_writer.add_scalar("epoch/train_loss", train_stats["loss"], epoch)

            lr_scheduler.step(epoch)

            if args.output_dir:
                if (epoch+1) % args.save_ep_freq == 0:
                    checkpoint_model = model.state_dict()
                    for k in model.state_dict().keys():
                        if 'token_mask' in k or 'adaptorFFN' in k or 'head' in  k:
                            continue
                        else:
                            del checkpoint_model[k]
                    checkpoint_paths = [output_dir / 'checkpoints/checkpoint-{}.pth'.format(epoch)]
                    for checkpoint_path in checkpoint_paths:
                        if model_ema is not None:
                            utils.save_on_master({
                                'model': checkpoint_model,
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'model_ema': get_state_dict(model_ema),
                                'scaler': loss_scaler.state_dict(),
                                'args': args,
                            }, checkpoint_path)
                        else:
                            utils.save_on_master({
                                'model': checkpoint_model,
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'scaler': loss_scaler.state_dict(),
                                'args': args,
                            }, checkpoint_path)

            test_stats = evaluate(data_loader=data_loader_val, model=model, model_name=args.model,
                                device=device, distillation_type=args.distillation_type,
                                args=args, task=t, approach=args.approach,)
            logger.info(test_stats)

            tb_writer.add_scalar("epoch/val_acc1", test_stats['acc1'], epoch)
            tb_writer.add_scalar("epoch/val_loss", test_stats['loss'], epoch)
            tb_writer.add_scalar("epoch/val_acc5", test_stats['acc5'], epoch)

            logger.info(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            logger.info(log_stats)

        logger.info("evaluation on old task")
        for u, uncla in taskcla:
            logger.info(f"evaluation on {dataloaders[u]['name']}")
            data_loader_val = dataloaders[u]['val']
            test_stats = evaluate(data_loader_val, model, 
                model_name=args.model, task=u, 
                device=device, distillation_type=args.distillation_type,
                args=args, approach=args.approach)
            logger.info(f"Accuracy of the network on {dataloaders[u]['name']}: \
                {test_stats['acc1']:.1f}%")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    __global_values__ = dict(it=0)

    main(args)
