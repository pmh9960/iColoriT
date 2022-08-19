import torch
import torch.nn as nn


class HintPropagationRange(nn.Module):
    '''
    Measure Hint Propagation Range
    https://arxiv.org/abs/2207.06831

    Just-Noticeable-Difference
    MacAdam, David L. "Visual sensitivities to color differences in daylight." Josa 32.5 (1942): 247-274.
    '''

    def __init__(self, jnd=2.3):
        super().__init__()
        self.prev_img = None
        self.jnd = jnd  # just-noticeable-difference

    def forward(self, pred, gt, hint_coord):
        diff_map, self.prev_img = self.get_diff_map_lab(self.prev_img, pred)
        changed_points = torch.stack(torch.where(torch.abs(diff_map) > self.jnd), dim=-1).float()  # N, 3
        dist_mean = 0
        dist_std = 0
        for idx in range(len(gt)):
            points = changed_points[changed_points[:, 0] == idx][:, 1:]  # n, 2 (x, y)
            if len(points) != 0:
                euc_dist = torch.sqrt(((points - hint_coord[idx].unsqueeze(0)) ** 2).sum(1))
                dist_mean += euc_dist.mean(0).item()
                if len(points) > 1:
                    dist_std += euc_dist.std(0).item()
        dist_mean, dist_std = dist_mean / len(gt), dist_std / len(gt)
        return dist_mean, dist_std

    def flush(self):
        self.prev_img = None

    @staticmethod
    def get_diff_map_lab(prev_img, pred_img):
        scale_factor = torch.tensor([100, 110, 110], device=pred_img.device)[None, :, None, None]
        diff_map = torch.zeros_like(pred_img)[:, 0, :, :]

        if prev_img is not None:
            diff_map = torch.sqrt((((pred_img - prev_img) * scale_factor)**2).sum(1))
        prev_img = pred_img.clone()

        return diff_map, prev_img
