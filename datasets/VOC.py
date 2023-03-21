import os
from typing import Optional, Iterable, Callable, Any, List, Tuple, Dict, cast
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
import torch
import torchvision
from PIL import Image, ImageDraw
from datasets.data_utils import get_anno_stats
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.common_types import _size_2_t
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

"""
This is the dataset module corresponds to torchvision==0.6.0, as on erdos cluster
"""


class VOCDataset(torchvision.datasets.VisionDataset):

    def __init__(self,
                 root: str,
                 anno_root: str,
                 cls_to_use: Optional[Iterable[str]] = None,
                 num_classes: Optional[int] = None,
                 object_only: bool = False,
                 transform: Optional[Callable] = None,
                 per_size: _size_2_t = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 ):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.cls_to_use = cls_to_use
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.num_classes = num_classes
        self.anno_root = anno_root
        self.per_size = per_size
        self.object_only = object_only
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS if is_valid_file is None else None,
                                    is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        assert num_classes is None or len(self.classes) == self.num_classes, "Found n_classes doesn't match given n_classes"
        self.num_classes = len(classes)



    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, bndbox, target = self.samples[index]

        if self.loader:
            sample = self.loader(path)
        else:
            sample = Image.open(path)
        if self.per_size or self.object_only:
            bndbox = sample.crop(bndbox)
        if self.transform is not None:
            if self.per_size or self.object_only:
                bndbox = self.transform(bndbox)
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.per_size:
            return sample, bndbox, target
        elif self.object_only:
            return bndbox, target
        else:
            return sample, target

    def _find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Modified version of torchvision.datasets.folder.find_classes
        to select a subset of image classes
        """
        if self.cls_to_use is not None:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name in self.cls_to_use)
        else:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir())
            if self.num_classes:
                classes = classes[:self.num_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
            self,
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, Tuple[int, int, int, int], int]]:
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self._find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            count = 0
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        img_name = fname.split('.')[0] # without .jpg
                        if self.per_size or self.object_only:
                            x_min, y_min, x_max, y_max, img_w, img_h = get_anno_stats(os.path.join(self.anno_root,
                                                                                                         target_class,
                                                                                                         img_name+'.xml'))
                            item = path, (x_min, y_min, x_max, y_max), class_index
                        else:
                            item = path, torch.tensor(0), class_index
                        instances.append(item)
                        count += 1

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances



if __name__ == '__main__':



    dir = '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered/train'

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)), torchvision.transforms.ToTensor()])

    dataset = VOCDataset(root=os.path.join(dir,'root'), anno_root=os.path.join(dir, 'annotations'), transform=transform, per_size=None)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    for img, bb, target in dataloader:
        t = torchvision.transforms.ToPILImage()
        t(img[0]).show()
        print(target)
