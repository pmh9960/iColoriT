# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os

import torch
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize, Resize

from dataset_folder import ImageFolder, ImageWithFixedHint, ImageWithFixedHintAndCoord
from hint_generator import InteractiveHintGenerator, RandomHintGenerator


class DataAugmentationForIColoriT:
    def __init__(self, args):
        # No normalization on RGB space
        mean = [0., 0., 0.]
        std = [1., 1., 1.]

        self.transform = Compose([
            RandomResizedCrop(args.input_size),
            ToTensor(),
            Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.hint_generator = RandomHintGenerator(args.input_size, args.hint_size, args.num_hint_range)

    def __call__(self, image):
        return self.transform(image), self.hint_generator()

    def __repr__(self):
        repr = "(DataAugmentationForIColoriT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Hint generator = %s,\n" % str(self.hint_generator)
        repr += ")"
        return repr


class DataTransformationForIColoriT:
    def __init__(self, args):

        self.transform = Compose([
            Resize((args.input_size, args.input_size)),
            ToTensor(),
        ])

        if args.hint_generator == 'RandomHintGenerator':
            self.hint_generator = RandomHintGenerator(args.input_size, args.hint_size, args.num_hint_range)
        elif args.hint_generator == 'InteractiveHintGenerator':
            self.hint_generator = InteractiveHintGenerator(args.input_size, args.hint_size)
        else:
            raise NotImplementedError(f'{args.hint_generator} is not exist.')

    def __call__(self, image):
        return self.transform(image), self.hint_generator()

    def __repr__(self):
        repr = "(DataTransformationForIColoriT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Hint generator = %s,\n" % str(self.hint_generator)
        repr += ")"
        return repr


class DataTransformationFixedHint:
    def __init__(self, args) -> None:
        self.input_size = args.input_size
        self.hint_size = args.hint_size
        self.img_transform = Compose([
            Resize((self.input_size, self.input_size)),
            ToTensor(),
        ])
        hint_dirs = args.hint_dirs
        if isinstance(args.hint_dirs, str):
            hint_dirs = [args.hint_dirs]
        self.num_hint = [int(os.path.basename(hint_dir)[4:])
                         for hint_dir in hint_dirs]  # hint subdir should be formed h#-n##

    def __call__(self, img, hint_coords):
        return self.img_transform(img), self.coord2hint(hint_coords)

    def coord2hint(self, hint_coords):
        hint = torch.ones((len(hint_coords), self.input_size // self.hint_size, self.input_size // self.hint_size))
        for idx, hint_coord in enumerate(hint_coords):
            for x, y in hint_coord:
                hint[idx, x // self.hint_size, y // self.hint_size] = 0
        return hint

    def __repr__(self):
        repr = "(DataTransformationFixedHint,\n"
        repr += "  img_transform = %s,\n" % str(self.img_transform)
        repr += f"  Hint generator = Fixed, {self.num_hint}\n"
        repr += ")"
        return repr


class DataTransformationFixedHintContinuousCoords:
    def __init__(self, args) -> None:
        self.input_size = args.input_size
        self.hint_size = args.hint_size
        self.img_transform = Compose([
            Resize((self.input_size, self.input_size)),
            ToTensor(),
        ])
        hint_dirs = args.hint_dirs
        if isinstance(args.hint_dirs, str):
            hint_dirs = [args.hint_dirs]
        self.num_hint = [int(os.path.basename(hint_dir).split(':')[-1]) for hint_dir in hint_dirs]

    def __call__(self, img, hint_coords):
        hint_coords = [hint_coords[0][:idx] for idx in range(len(hint_coords[0]) + 1)]
        return self.img_transform(img), self.coord2hint(hint_coords)

    def coord2hint(self, hint_coords):
        hint = torch.ones((len(hint_coords), self.input_size // self.hint_size, self.input_size // self.hint_size))
        for idx, hint_coord in enumerate(hint_coords):
            for x, y in hint_coord:
                hint[idx, x // self.hint_size, y // self.hint_size] = 0
        return hint

    def __repr__(self):
        repr = "(DataTransformationFixedHint,\n"
        repr += "  img_transform = %s,\n" % str(self.img_transform)
        repr += f"  Hint generator = Fixed, {self.num_hint}\n"
        repr += ")"
        return repr


class DataTransformationFixedHintPrevCoods:
    def __init__(self, args) -> None:
        self.input_size = args.input_size
        self.hint_size = args.hint_size
        self.img_transform = Compose([
            Resize((self.input_size, self.input_size)),
            ToTensor(),
        ])
        hint_dirs = args.hint_dirs
        if isinstance(args.hint_dirs, str):
            hint_dirs = [args.hint_dirs]
        self.num_hint = [int(os.path.basename(hint_dir).split(':')[-1]) for hint_dir in hint_dirs]

    def __call__(self, img, hint_coords):
        hint_coords = [[hint_coord[:-1], hint_coord] for hint_coord in hint_coords]
        return self.img_transform(img), self.coord2hint_prev(hint_coords)

    def coord2hint_prev(self, hint_coords):
        hint = torch.ones((len(hint_coords), 2, self.input_size // self.hint_size, self.input_size // self.hint_size))
        for idx, (hint_coord_prev, hint_coord) in enumerate(hint_coords):
            for x, y in hint_coord_prev:
                hint[idx, 0, x // self.hint_size, y // self.hint_size] = 0
            for x, y in hint_coord:
                hint[idx, 1, x // self.hint_size, y // self.hint_size] = 0
        return hint

    def __repr__(self):
        repr = "(DataTransformationFixedHint,\n"
        repr += "  img_transform = %s,\n" % str(self.img_transform)
        repr += f"  Hint generator = Fixed, {self.num_hint}\n"
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForIColoriT(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)


def build_validation_dataset(args):
    transform = DataTransformationForIColoriT(args)
    print("Data Trans = %s" % str(transform))
    return ImageFolder(args.val_data_path, transform=transform,
                       is_valid_file=(lambda x: False if '.pt' in x else True))


def build_fixed_validation_dataset(args):
    transform = DataTransformationFixedHint(args)
    print("Data Trans = %s" % str(transform))
    return ImageWithFixedHint(args.val_data_path, args.hint_dirs, transform=transform,
                              return_name=args.return_name, gray_file_list_txt=args.gray_file_list_txt)


def build_fixed_validation_dataset_coord(args, without_tf=False):
    transform = DataTransformationFixedHintContinuousCoords(args) if not without_tf else None
    print("Data Trans = %s" % str(transform))
    return ImageWithFixedHintAndCoord(args.val_data_path, args.hint_dirs, transform=transform)


def build_fixed_validation_dataset_coord_2(args, without_tf=False):
    transform = DataTransformationFixedHintPrevCoods(args) if not without_tf else None
    print("Data Trans = %s" % str(transform))
    return ImageWithFixedHintAndCoord(args.val_data_path, args.hint_dirs, transform=transform)
