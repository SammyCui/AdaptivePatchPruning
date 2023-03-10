"""

Ref: https://github.com/youweiliang/evit

"""


import os
import torch
from timm import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from einops import rearrange
from torchvision import transforms
import models.evit
from datasets.VOC import VOCDataset


def mask(x, idx, patch_size):
    """
    Args:
        x: input image, shape: [B, 3, H, W]
        idx: indices of masks, shape: [B, T], value in range [0, h*w)
    Return:
        out_img: masked image with only patches from idx postions
    """
    h = x.size(2) // patch_size
    x = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=patch_size, q=patch_size)
    output = torch.zeros_like(x)
    idx1 = idx.unsqueeze(1).expand(-1, x.size(1), -1)
    extracted = torch.gather(x, dim=2, index=idx1)  # [b, c p q, T]
    scattered = torch.scatter(output, dim=2, index=idx1, src=extracted)
    out_img = rearrange(scattered, 'b (c p q) (h w) -> b c (h p) (w q)', p=patch_size, q=patch_size, h=h)
    return out_img


def get_deeper_idx(idx1, idx2):
    """
    Args:
        idx1: indices, shape: [B, T1]
        idx2: indices to gather from idx1, shape: [B, T2], T2 <= T1
    """
    return torch.gather(idx1, dim=1, index=idx2)


def get_real_idx(idxs, fuse_token):
    # nh = img_size // patch_size
    # npatch = nh ** 2

    # gather real idx
    for i in range(1, len(idxs)):
        tmp = idxs[i - 1]
        if fuse_token:
            B = tmp.size(0)
            tmp = torch.cat([tmp, torch.zeros(B, 1, dtype=tmp.dtype, device=tmp.device)], dim=1)
        idxs[i] = torch.gather(tmp, dim=1, index=idxs[i])
    return idxs


def save_img_batch(x, path, file_name='img{}', start_idx=0):
    for i, img in enumerate(x):
        save_image(img, os.path.join(path, file_name.format(start_idx + i)))

@torch.no_grad()
def visualize(model, dataloader, keep_rate, fuse_token, device):
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)

    # switch to evaluation mode
    model.eval()

    ii = 0
    for images, target in dataloader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        B = images.size(0)

        with torch.cuda.amp.autocast():
            output, idx = model(images, keep_rate, get_idx=True)

        # denormalize
        images = images * std + mean
        concat_img = torch.tensor([])
        idxs = get_real_idx(idx, fuse_token)
        for jj, idx in enumerate(idxs):
            masked_img = mask(images, patch_size=16, idx=idx)

            row = rearrange(masked_img, 'b c h w -> c h (b w)')
            concat_img = torch.cat((concat_img, row),dim=1)
        t = transforms.ToPILImage()
        concat_img = t(concat_img)
        concat_img.show()


if __name__ == '__main__':
    model_default_pretrained = create_model('deit_small_patch16_224_shrink_base', pretrained=True)
    model_evit_pretrained = create_model('deit_small_patch16_224_shrink_base', pretrained=False)
    model_evit_pretrained.load_state_dict(torch.load('/Users/xuanmingcui/Downloads/evit-0.7-fuse-img224-deit-s.pth',
                                                     map_location='cpu')['model'])
    train_root = '/Users/xuanmingcui/Documents/cnslab/VOC2012_filtered/train/root'
    train_anno = '/Users/xuanmingcui/Documents/cnslab/VOC2012_filtered/train/annotations'
    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                                    ])

    dataset=VOCDataset(root=train_root, anno_root=train_anno, transform=transform)
    dataloader = DataLoader(dataset,  batch_size=4, num_workers=1, shuffle=False)
    visualize(model_evit_pretrained, dataloader, keep_rate=.7, fuse_token=False, device='cpu')




