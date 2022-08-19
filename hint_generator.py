# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import copy

import numpy as np
import torch


class RandomHintGenerator:
    '''
    Use RandomHintGenerator in BEiT as random hint generator
    '''

    def __init__(self, input_size, hint_size=2, num_hint_range=[10, 10]):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_hint_location = self.height * self.width // (hint_size * hint_size)
        self.num_hint_range = num_hint_range

    def __repr__(self):
        repr_str = (f'Hint: total hint locations {self.num_hint_location},'
                    f'number of hints range {self.num_hint_range}')
        return repr_str

    def __call__(self):
        return self.uniform_gen()

    def uniform_gen(self):
        num_hint = np.random.random_integers(self.num_hint_range[0], self.num_hint_range[1])
        hint = np.hstack([
            np.ones(self.num_hint_location - num_hint),
            np.zeros(num_hint),
        ])
        np.random.shuffle(hint)
        return hint


class InteractiveHintGenerator:
    ''' Interactive hint generator by user input '''

    def __init__(self, input_size, hint_size):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.hint_size = hint_size

        # set hyper-parameters
        self.height, self.width = input_size
        self.hint_size = hint_size
        self.num_hint_location = self.height * self.width // (hint_size * hint_size)

        self.hint = np.ones((self.height // hint_size, self.width // hint_size))
        self.coord_xs, self.coord_ys = None, None

    def __repr__(self):
        repr_str = f"Hint: total hint locations {self.num_hint_location}"
        return repr_str

    def __call__(self):
        if self.coord_xs is None:
            self.coord_xs, self.coord_ys = [], []
            return copy.deepcopy(self.hint), torch.tensor((self.coord_xs, self.coord_ys)).T
        while True:
            coord_x = float(input('coord_x: '))
            coord_y = float(input('coord_y: '))
            if coord_x >= 0 and coord_y >= 0 and coord_x < self.height and coord_y < self.width:
                break
            print(f'coord_x, coord_y should be in [0, {self.height}) [0, {self.width})')

        self.coord_xs.append(coord_x)
        self.coord_ys.append(coord_y)
        coord_x = int(coord_x // self.hint_size)
        coord_y = int(coord_y // self.hint_size)
        coord_x = self.hint.shape[0] - 1 if coord_x >= self.hint.shape[0] else coord_x
        coord_y = self.hint.shape[1] - 1 if coord_y >= self.hint.shape[1] else coord_y

        self.hint[coord_x, coord_y] = 0

        return copy.deepcopy(self.hint), torch.tensor((self.coord_xs, self.coord_ys)).T
