import einops
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timeit import default_timer as timer
from utils.visualizations import get_real_idx, mask
from .base import BaseTrainer
import torch
from torchvision.utils import save_image
from einops import rearrange
from torchvision import transforms
from scipy.stats.kde import gaussian_kde
from numpy import linspace
from matplotlib import pyplot as plt
import seaborn as sb
import scipy


class adavitTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)


    def train(self):
        print('==> Training Start')

        sigma = self.args.sigma
        for epoch in range(self.start_epoch, self.max_epoch + 1):

            epoch_t0 = timer()
            if 'perturb' in  self.args.model:
                if self.args.decay_sigma:
                    sigma = (1 - epoch / self.max_epoch) * self.args.sigma
                self._train_one_epoch(sigma=sigma)
            else:
                self._train_one_epoch()
            self.train_time.update(timer() - epoch_t0)


    @torch.no_grad()
    def visualize(self):
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=self.device).reshape(3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD, device=self.device).reshape(3, 1, 1)

        # switch to evaluation mode
        self.model.eval()
        save_n_batch = self.args.save_n_batch

        for images, target in self.test_dataloader:

            if save_n_batch > 0:
                images = images.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    output, idx = self.model(images, self.args.base_keep_rate, get_idx=True)

                # denormalize
                images = images * std + mean
                concat_img = torch.tensor([])
                idxs = get_real_idx(idx, self.args.fuse_token)
                for jj, idx in enumerate(idxs):
                    masked_img = mask(images, patch_size=16, idx=idx)

                    row = rearrange(masked_img, 'b c h w -> c h (b w)')
                    concat_img = torch.cat((concat_img, row), dim=1)
                t = transforms.ToPILImage()
                concat_img = t(concat_img)
                save_image(concat_img, self.args.result_dir)
                save_n_batch -= 1
            else:
                break

    def plot_attn_dist(self, mode, log_scale=False):
        self.model.eval()
        save_n_batch = self.args.save_n_batch

        total_layers_attns = None

        for images, target in self.test_dataloader:
            if save_n_batch > 0:
                images = images.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    output, img_attns = self.model(images, get_img_attn = True)
                    img_attns = np.array([layer.detach().numpy() for layer in img_attns]) # 12 B N
                    # img_attns = [einops.rearrange(img_attn, "B N -> (B N)") for img_attn in img_attns]
                    # if log_scale:
                    #     img_attns = np.array([torch.log(layer).detach().numpy() for layer in img_attns])
                    # else:
                    #     img_attns = np.array([layer.detach().numpy() for layer in img_attns])

                # denormalize
                mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=self.device).reshape(3, 1, 1)
                std = torch.tensor(IMAGENET_DEFAULT_STD, device=self.device).reshape(3, 1, 1)
                images = images * std + mean

                if  total_layers_attns is None:
                    total_layers_attns = img_attns
                else:
                    total_layers_attns = np.append(total_layers_attns, img_attns, axis=1)

                save_n_batch -= 1
            else:
                break

        for m in mode:
            for idx, layer in enumerate(total_layers_attns):
                if idx % 3 != 0:
                    continue

                if m =='dist':
                    img_attn = einops.rearrange(layer, "B N -> (B N)")
                    if log_scale:
                        img_attn = np.log(img_attn)

                    print(f"layer {idx},  mean={np.mean(img_attn, axis=-1)}, median={np.median(img_attn, axis=-1)}, "
                          f"skewness={scipy.stats.skew(img_attn, axis=-1)}, kurtosis={scipy.stats.kurtosis(img_attn, axis=-1)}")
                    sb.histplot(img_attn, element='poly', fill=False, label=idx)
                elif m =='rank':
                    img_attn = np.sort(layer, axis=-1).mean(axis=0)
                    plt.plot(img_attn, label=idx)

            plt.legend()
            plt.show()

