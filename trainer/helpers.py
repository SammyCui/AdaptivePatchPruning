import argparse
import os
from typing import List
import re
import torch
from timm import create_model
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets.VOC import VOCDataset

DEFAULT_ROOT = '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/'
DEFAULT_CLS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_model_optimizer(args):


    if 'adavit' in args.model:
        model = create_model(args.model,
                             pretrained=args.pretrained,
                             )
    else:
        model = create_model(
            args.model,
            base_keep_rate=args.base_keep_rate,
            drop_loc=args.drop_loc,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            fuse_token=args.fuse_token,
            img_size=(args.image_size, args.image_size)
        )

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    else:
        raise Exception("Unknown optimizer")

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

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['models'])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['train_epoch']
        args.start_step = checkpoint['train_step']

    return model, optimizer, lr_scheduler

def get_dataloaders(args):
    transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    # train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)
    # val_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True)
    train_dataset = VOCDataset(root=os.path.join(args.train_root, 'root'), anno_root=os.path.join(args.train_root, 'annotations'),
                               cls_to_use=args.cls_to_use,
                               transform=transform,
                               per_size=args.per_size,
                               object_only=args.object_only)
    val_dataset = VOCDataset(root=os.path.join(args.val_root, 'root'), anno_root=os.path.join(args.val_root, 'annotations'),
                            cls_to_use=args.cls_to_use,
                            transform=transform,
                            per_size=args.per_size,
                            object_only=args.object_only)
    test_dataset = VOCDataset(root=os.path.join(args.test_root, 'root'), anno_root=os.path.join(args.test_root, 'annotations'),
                              cls_to_use=args.cls_to_use,
                              transform=transform,
                              per_size=args.per_size,
                              object_only=args.object_only)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader



def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--root', type=str, nargs='?', const=DEFAULT_ROOT, default=DEFAULT_ROOT)

    parser.add_argument('--train', type=str, default='True')
    parser.add_argument('--model', type=str, nargs='?', const='Benchmark', default='Benchmark')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--subset_data', type=str, default='False')

    parser.add_argument('--base_keep_rate', type=float, default=0.8,
                        help="Fixed keep ratio for evit")
    parser.add_argument('--drop_loc', default='(3, 6, 9)', type=str,
                        help='the layer indices for shrinking inattentive tokens')
    parser.add_argument('--fuse_token', action='store_true',
                        help='whether to fuse the inattentive tokens')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # training params.
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['multistep', 'step', 'cosine'], nargs='?', const=None)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.2)  # for lr_scheduler
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='./u/erdos/cnslab/xcui32/AdaptivePatchSelectionViT/results')
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
    args.train = eval(args.train)
    args.save = eval(args.save)
    args.subset_data = eval(args.subset_data)
    args.drop_loc = eval(args.drop_loc)
    # if subset data, take 10 default classes
    if args.subset_data:
        args.num_classes = 10
        args.cls_to_use = DEFAULT_CLS
    else:
        args.num_classes = 20
        args.cls_to_use = None
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
                 step_size: int = 20,
                 gamma: float = 0.2,
                 num_classes: int =20,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 train: bool = True,
                 val_interval: int = 1,
                 num_workers: int = 1,
                 batch_size: int = 2,
                 pretrained: bool = True,
                 download: bool = False,
                 device: str = 'cpu',
                 result_dir: str = './results',
                 save: bool = False,
                 resume: bool = False,
                 init_backbone: bool = False):

        self.root = root
        self.cls_to_use = cls_to_use
        self.keep_ratio = keep_ratio
        self.patch_size = patch_size
        self.write_to_collections = write_to_collections
        self.run_name = run_name
        self.image_size = image_size
        self.model = model
        self.download = download
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.lr = lr
        self.train = train
        self.optimizer = optimizer
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
