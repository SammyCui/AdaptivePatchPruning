import argparse
import os
from typing import List
import re

import timm.data
import torch
from timm import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from models import adavit, evit
from datasets.VOC import VOCDataset

DEFAULT_ROOT = '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/'
DEFAULT_CLS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']


def get_model_optimizer(args):
    if 'ada' in args.model:
        model = create_model(args.model,
                             pretrained=args.pretrained,
                             drop_rate=args.drop,
                             drop_path_rate=args.drop_path,
                             drop_block_rate=None,
                             img_size=(args.image_size, args.image_size),
                             keep_rate=args.keep_rate,
                             prune_loc=args.prune_loc,
                             num_classes=args.num_classes
                             )
    elif 'evit' in args.model or 'shrink' in args.model:
        model = create_model(
            args.model,
            base_keep_rate=args.keep_rate,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            fuse_token=args.fuse_token,
            img_size=(args.image_size, args.image_size)
        )

    else:
        raise NotImplementedError("Model not implemented")

    if args.optimizer == 'sgd':
        if args.pretrained:
            optimizer = optim.SGD([{'params': model.head.parameters(), 'lr': args.lr * 100}],
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        if args.pretrained:
            optimizer = optim.Adam([{'params': model.head.parameters(), 'lr': args.lr * 100}], lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

    else:
        raise NotImplementedError("Unknown optimizer")

    if args.lr_scheduler:
        if args.lr_scheduler == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(args.step_size),
                gamma=args.gamma
            )
        elif args.lr_scheduler == 'multistep':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(_) for _ in args.step_size.split(',')],
                gamma=args.gamma,
            )
        elif args.lr_scheduler == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                args.max_epoch,
                eta_min=0  # a tuning parameter
            )
        else:
            raise ValueError('Unknown Scheduler')
    else:
        lr_scheduler = None

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['models'])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['train_epoch']
        args.start_step = checkpoint['train_step']

    return model, criterion, optimizer, lr_scheduler


def build_transform(is_train, args):
    resize_im = args.image_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.image_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.image_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_dataloaders(args):
    train_transform = build_transform(is_train=True, args=args)
    val_transform = build_transform(is_train=True, args=args)

    # train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)
    # val_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True)
    train_dataset = VOCDataset(root=os.path.join(args.train_root, 'root'),
                               anno_root=os.path.join(args.train_root, 'annotations'),
                               transform=train_transform)
    val_dataset = VOCDataset(root=os.path.join(args.val_root, 'root'),
                             anno_root=os.path.join(args.val_root, 'annotations'),
                             transform=val_transform)
    test_dataset = VOCDataset(root=os.path.join(args.test_root, 'root'),
                              anno_root=os.path.join(args.test_root, 'annotations'),
                              transform=val_transform)

    train_num_classes = train_dataset.num_classes
    val_num_classes = val_dataset.num_classes
    test_num_classes = test_dataset.num_classes

    assert train_num_classes == val_num_classes == test_num_classes, "num_classes between train, val and test don't match"

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader, train_num_classes


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--root', type=str, nargs='?', const=DEFAULT_ROOT, default=DEFAULT_ROOT)

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'visualize', 'plot_attn_dist'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_n_batch', type=int, default=1,
                        help="how many batch to save for visualize and plot_attn_dist operations")

    parser.add_argument('--keep_rate', type=float, default=0.7,
                        help="Fixed keep ratio for evit")
    parser.add_argument('--prune_loc', default='(3, 6, 9)', type=str,
                        help='the layer indices for patch pruning')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help="Starting sigma for perturbed optimizer.")
    parser.add_argument('--decay_sigma', type=str, default='True',
                        help="Linearly decay sigma to 0 over epochs for perturbed optimizer.")
    parser.add_argument('--fuse_token', action='store_true',
                        help='whether to fuse the inattentive tokens')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')

    # training params.
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['multistep', 'step', 'cosine'], nargs='?',
                        const=None)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.2)  # for lr_scheduler
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # data and augmentations
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                               "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
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

    # Mixup params
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

    # training devices and logging
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='./u/erdos/cnslab/xcui32/AdaptivePatchPruning/results')
    parser.add_argument('--download', type=str, default='False', nargs='?', const='False',
                        help='for torchvision dataset download param')
    parser.add_argument('--save', type=str, default='False', nargs='?', const='False')
    parser.add_argument('--write_to_collections', type=str, default=None, nargs='?', const=None,
                        help='write training summary to specified file')
    parser.add_argument('--resume', type=str, default=None, nargs='?', const=None)
    parser.add_argument('--pretrained', type=str, default='False')

    args = parser.parse_args()
    args = post_process_args(args)
    return args


def post_process_args(args):
    args.train_root = os.path.join(args.root, 'train')
    args.val_root = os.path.join(args.root, 'val')
    args.test_root = os.path.join(args.root, 'test')
    args.pretrained = eval(args.pretrained)
    args.save = eval(args.save)
    args.prune_loc = eval(args.prune_loc)
    if args.mode == 'plot_attn_dist':
        args.get_img_attns = True
    if args.device == 'gpu':
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


class DebugArgs:
    def __init__(self,

                 root: str = '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered',
                 model: str = 'FE_WeightShare',
                 start_epoch: int = 0,
                 max_epoch: int = 200,
                 lr: float = 0.001,
                 optimizer: str = 'adam',
                 lr_scheduler: str = 'step',
                 write_to_collections: str = None,
                 run_name: str = None,
                 image_size: int = 224,
                 cls_to_use: List[str] = None,
                 step_size: int = 20,
                 gamma: float = 0.2,
                 mode: str = 'plot_attn_dist',
                 sigma: float = 0.1,
                 decay_sigma: bool = True,
                 num_classes: int = 20,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 train: bool = True,
                 prune_loc=(3, 6, 9),
                 val_interval: int = 1,
                 keep_rate: float = 1,
                 num_workers: int = 1,
                 save_n_batch: int = 1,
                 color_jitter=0.4,
                 aa='rand-m9-mstd0.5-inc1',
                 smoothing=0.1,
                 train_interpolation='bicubic',
                 repeated_aug=True,
                 no_repeated_aug=False,
                 reprob=0.25,
                 remode='pixel',
                 recount=1,
                 resplit=False,
                 mixup=0.8,
                 cutmix=1,
                 cutmix_minmax=None,
                 mixup_prob=1,
                 mixup_switch_prob=0.5,
                 mixup_mode='batch',
                 batch_size: int = 2,
                 drop_path: float = 0.1,
                 fuse_token: bool = True,
                 pretrained: bool = True,
                 download: bool = False,
                 device: str = 'cpu',
                 drop: float = 0,
                 result_dir: str = './results',
                 save: bool = False,
                 resume: bool = False,
                 init_backbone: bool = False):
        self.root = root
        self.write_to_collections = write_to_collections
        self.run_name = run_name
        self.image_size = image_size
        self.model = model
        self.color_jitter = color_jitter
        self.aa = aa
        self.smoothing = smoothing
        self.train_interpolation = train_interpolation
        self.repeated_aug = repeated_aug
        self.no_repeated_aug = no_repeated_aug
        self.reprob = reprob
        self.remode = remode
        self.recount = recount
        self.resplit = resplit
        self.mixup = mixup
        self.cutmix = cutmix
        self.cutmix_minmax = cutmix_minmax
        self.mixup_prob = mixup_prob
        self.mixup_switch_prob = mixup_switch_prob
        self.mixup_mode = mixup_mode
        self.save_n_batch = save_n_batch
        self.drop = drop
        self.download = download
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.start_epoch = start_epoch
        self.sigma = sigma
        self.decay_sigma = decay_sigma
        self.fuse_token = fuse_token
        self.max_epoch = max_epoch
        self.lr = lr
        self.drop_path = drop_path
        self.keep_rate = keep_rate
        self.prune_loc = prune_loc
        self.mode = mode
        self.train = train
        self.optimizer = optimizer
        self.cls_to_use = cls_to_use
        self.lr_scheduler = lr_scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.momentum = momentum
        self.pretrained = pretrained
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_interval = val_interval
        self.device = device
        self.result_dir = result_dir
        self.save = save
        self.resume = resume
        self.init_backbone = init_backbone

        self.train_root = os.path.join(self.root, 'train')
        self.val_root = os.path.join(self.root, 'val')
        self.test_root = os.path.join(self.root, 'test')


if __name__ == '__main__':
    a = create_model('deit_base_patch16_384_test')
