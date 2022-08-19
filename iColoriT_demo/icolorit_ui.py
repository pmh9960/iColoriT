import sys
import os
import argparse

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from timm.models import create_model
import torch

from gui import gui_main
import modeling


def get_args():
    parser = argparse.ArgumentParser('Colorization UI', add_help=False)
    # Directories
    parser.add_argument('--model_path', type=str, default='path/to/checkpoints', help='checkpoint path of model')
    parser.add_argument('--target_image', default='path/to/image', type=str, help='validation dataset path')
    parser.add_argument('--device', default='cpu', help='device to use for testing')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')

    # Model parameters
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode')
    parser.add_argument('--mask_cent', action='store_true', help='mask_cent')

    args = parser.parse_args()

    return args


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        mask_cent=args.mask_cent,
    )

    return model


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()

    model = get_model(args)
    model.to(args.device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    app = QApplication(sys.argv)
    ex = gui_main.IColoriTUI(color_model=model, img_file=args.target_image,
                             load_size=args.input_size, win_size=720, device=args.device)
    ex.setWindowIcon(QIcon('gui/icon.png'))
    ex.setWindowTitle('iColoriT')
    ex.show()
    sys.exit(app.exec_())
