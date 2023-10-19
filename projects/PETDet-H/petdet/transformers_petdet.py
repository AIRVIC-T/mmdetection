# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes


@TRANSFORMS.register_module()
class BoxJitter(BaseTransform):
    def __init__(self, amplitude: float = 0.05):
        self.amplitude = amplitude

    def transform(self, results: dict) -> dict:
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            if isinstance(gt_bboxes, np.ndarray):
                random_offsets = np.random.uniform(
                    -self.amplitude, self.amplitude, (gt_bboxes.shape[0], 4))
                # before jittering
                cxcy = (gt_bboxes[:, 2:4] + gt_bboxes[:, :2]) / 2
                wh = np.abs(gt_bboxes[:, 2:4] - gt_bboxes[:, :2])
                # after jittering
                new_cxcy = cxcy + wh * random_offsets[:, :2]
                new_wh = wh * (1 + random_offsets[:, 2:])
                # xywh to xyxy
                new_x1y1 = (new_cxcy - new_wh / 2)
                new_x2y2 = (new_cxcy + new_wh / 2)
                gt_bboxes = np.concatenate([new_x1y1, new_x2y2], axis=1)
                results['gt_bboxes'] = gt_bboxes
            elif isinstance(gt_bboxes, HorizontalBoxes):
                gt_bboxes = results['gt_bboxes'].tensor
                random_offsets = gt_bboxes.new_empty(
                    gt_bboxes.shape[0], 4).uniform_(-self.amplitude, self.amplitude)
                # before jittering
                cxcy = (gt_bboxes[:, 2:4] + gt_bboxes[:, :2]) / 2
                wh = (gt_bboxes[:, 2:4] - gt_bboxes[:, :2]).abs()
                # after jittering
                new_cxcy = cxcy + wh * random_offsets[:, :2]
                new_wh = wh * (1 + random_offsets[:, 2:])
                # xywh to xyxy
                new_x1y1 = (new_cxcy - new_wh / 2)
                new_x2y2 = (new_cxcy + new_wh / 2)
                gt_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
                results['gt_bboxes'] = HorizontalBoxes(gt_bboxes)
            else:
                raise NotImplementedError
        return results
