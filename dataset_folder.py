# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import os.path as osp
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = osp.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = osp.join(directory, target_class)
        if not osp.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = osp.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)

        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        # saving samples.pkl can skip above line
        # with open('debug/samples.pkl', 'rb') as f:
        #     samples = pickle.load(f)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform, target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


class ImageWithFixedHint(Dataset):
    def __init__(self, root, hint_dirs, transform=None, return_name=False, gray_file_list_txt=''):
        super().__init__()
        self.img_dir = osp.join(root, 'imgs')
        if isinstance(hint_dirs, str):
            hint_dirs = [hint_dirs]
        for hint_dir in hint_dirs:
            if not osp.isdir(hint_dir):
                raise FileNotFoundError(f'{hint_dir} is not exist!')
        self.hint_dirs = hint_dirs
        self.transform = transform
        self.return_name = return_name

        # do not use gray images
        self.gray_imgs = []
        if gray_file_list_txt:
            with open(gray_file_list_txt, 'r') as f:
                self.gray_imgs = [osp.splitext(osp.basename(i))[0] for i in f.readlines()]

        self.img_list = [file for file in os.listdir(self.img_dir)
                         if is_image_file(file) and not self.is_gray(file)]
        self.img_list = sorted(self.img_list)
        for hint_dir in self.hint_dirs:
            self.hint_list = [file for file in os.listdir(hint_dir)
                              if file.endswith('.txt') and not self.is_gray(file)]
            self.hint_list = sorted(self.hint_list)

            # check name
            assert len(self.img_list) == len(self.hint_list)
            for img_f, hint_f in zip(self.img_list, self.hint_list):
                assert osp.splitext(img_f)[0] == osp.splitext(hint_f)[0]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # load image
        img_f = osp.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_f)

        # load_hint
        hint_coords = []
        for hint_dir in self.hint_dirs:
            hint_f = osp.join(hint_dir, self.hint_list[idx])
            with open(hint_f, 'r') as f:
                hint_coord = f.readlines()
                hint_coord = [file.strip().split(' ') for file in hint_coord]
                hint_coord = [(int(coord[0]), int(coord[1])) for coord in hint_coord]
            hint_coords.append(hint_coord)
        img, hint_coords = self.transform(img, hint_coords)
        target = 0
        if self.return_name:
            return (img, hint_coords), target, self.img_list[idx]
        return (img, hint_coords), target

    def is_gray(self, file):
        return osp.splitext(file)[0] in self.gray_imgs


class ImageWithFixedHintAndCoord(Dataset):
    def __init__(self, root, hint_dirs, transform=None):
        super().__init__()
        self.img_dir = osp.join(root, 'imgs')
        if isinstance(hint_dirs, str):
            hint_dirs = [hint_dirs]
        for hint_dir in hint_dirs:
            if not osp.isdir(hint_dir):
                raise FileNotFoundError(f'{hint_dir} is not exist!')
        self.hint_dirs = hint_dirs
        self.transform = transform

        self.img_list = [file for file in os.listdir(self.img_dir)
                         if is_image_file(file)]
        self.img_list = sorted(self.img_list)
        for hint_dir in self.hint_dirs:
            self.hint_list = [file for file in os.listdir(hint_dir)
                              if file.endswith('.txt')]
            self.hint_list = sorted(self.hint_list)

            # assertion
            assert len(self.img_list) == len(self.hint_list)
            for img_f, hint_f in zip(self.img_list, self.hint_list):
                assert osp.splitext(img_f)[0] == osp.splitext(hint_f)[0]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # load image
        img_f = osp.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_f)

        # load_hint
        hint_coords = []
        for hint_dir in self.hint_dirs:
            hint_f = osp.join(hint_dir, self.hint_list[idx])
            with open(hint_f, 'r') as f:
                hint_coord = f.readlines()
                hint_coord = [file.strip().split(' ') for file in hint_coord]
                hint_coord = [(int(coord[0]), int(coord[1]))
                              for coord in hint_coord]
            hint_coords.append(torch.tensor(hint_coord))
        img, hint = self.transform(img, hint_coords)
        target = 0
        return (img, hint, hint_coords), target
