import torch

from mmengine.model import bias_init_with_prob, normal_init

from mmdet.models.dense_heads.tood_head import TOODHead
from mmdet.registry import MODELS


@MODELS.register_module()
class TOODHeadWithScore(TOODHead):
    def __init__(self,
                 start_level=0,
                 out_score=False,
                 **kwargs) -> None:
        super().__init__(
            **kwargs)
        self.start_level = start_level
        self.out_score = out_score

    def init_weights(self):
        super().init_weights()
        bias_cls = bias_init_with_prob(0.05)
        normal_init(self.tood_cls, std=0.01, bias=bias_cls)

    def forward(self, feats):
        return super().forward(feats[self.start_level:])

    def _bbox_post_process(self, **kwargs):
        results = super()._bbox_post_process(**kwargs)
        if self.out_score:
            results.bboxes = torch.cat(
                [results.bboxes, results.scores[:, None]], dim=1)
        return results
