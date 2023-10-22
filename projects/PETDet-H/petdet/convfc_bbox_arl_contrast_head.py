import torch
import torch.nn as nn

from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.structures.bbox import get_box_tensor
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from typing import Optional

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.structures.bbox.horizontal_boxes import HorizontalBoxes


class Contrastivebranch(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """

    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )

    def init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feat = self.head(x)
        return feat


@MODELS.register_module()
class ConvFCBBoxARLContrastHead(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 contrast_out_channels=512,
                 loss_contrast=dict(
                     type='SupConProxyAnchorLoss',
                     class_num=20,
                     size_contrast=512,
                     stage=2,
                     mrg=0,
                     alpha=32,
                     loss_weight=0.5),
                 loss_iou=dict(type='GIoULoss', loss_weight=1.0),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.contrast_out_channels = contrast_out_channels
        self.loss_contrast = MODELS.build(loss_contrast)
        self.loss_iou = MODELS.build(loss_iou)
        self.encoder = Contrastivebranch(
            self.fc_out_channels, self.contrast_out_channels)  # self.mlp_head_dim 256 or 128

    def init_weights(self):
        super().init_weights()
        # conv layers are already initialized by ConvModule
        self.encoder.init_weights()

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        contrast_feature: Tensor,
                        rois: Tensor,
                        sampling_results,
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        losses = self.loss(
            cls_score,
            bbox_pred,
            contrast_feature,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        x_contrast = x_cls
        contrast_feature = self.encoder(x_contrast)
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, contrast_feature

    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim + reg_dim + 1)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bboxes_score = pos_priors[:, -1]
            pos_bbox_targets_encode = self.bbox_coder.encode(
                HorizontalBoxes(pos_priors[:, :-1]), pos_gt_bboxes)
            pos_bbox_targets_encode_unencode = get_box_tensor(pos_gt_bboxes)
            pos_bbox_targets = torch.cat(
                [pos_bbox_targets_encode_unencode, pos_bbox_targets_encode, pos_bboxes_score[:, None]], dim=1)
            # if not self.reg_decoded_bbox:
            #     pos_bbox_targets = self.bbox_coder.encode(
            #         HorizontalBoxes(pos_priors[:, :-1]), pos_gt_bboxes)
            #     pos_bbox_targets = torch.cat(
            #         [pos_bbox_targets, pos_bboxes_score[:, None]], dim=1)
            # else:
            #     # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            #     # is applied directly on the decoded bounding boxes, both
            #     # the predicted boxes and regression targets should be with
            #     # absolute coordinate format.
            #     pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            #     pos_bbox_targets = torch.cat(
            #         [pos_bbox_targets, pos_bboxes_score[:, None]], dim=1)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             contrast_feature: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        # loss_bbox
        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            bbox_pred_decode = self.bbox_coder.decode(rois[:, 1:-1], bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
            # if self.reg_decoded_bbox:
            #     # When the regression loss (e.g. `IouLoss`,
            #     # `GIouLoss`, `DIouLoss`) is applied directly on
            #     # the decoded bounding boxes, it decodes the
            #     # already encoded coordinates to absolute format.
            #     bbox_pred = self.bbox_coder.decode(rois[:, 1:-1], bbox_pred)
            #     bbox_pred = get_box_tensor(bbox_pred)
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                pos_bbox_pred_decode = bbox_pred_decode.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)]]
                pos_bbox_pred_decode = bbox_pred_decode.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)]]
            losses_bbox = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds.type(torch.bool), 4:-1],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
            losses_iou = self.loss_iou(
                pos_bbox_pred_decode,
                bbox_targets[pos_inds.type(torch.bool), :4],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses_bbox = bbox_pred[pos_inds].sum()
            losses_iou = bbox_pred[pos_inds].sum()

        # loss_cls
        if cls_score.numel() > 0:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            weight = torch.ones_like(labels).float()
            joint_weight = None
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                pos_decode_bbox_pred = self.bbox_coder.decode(
                    rois[pos_inds.type(torch.bool), 1:-1], pos_bbox_pred)
                pos_decode_bbox_target = self.bbox_coder.decode(
                    rois[pos_inds.type(torch.bool), 1:-1], bbox_targets[pos_inds.type(torch.bool), 4:-1])
                iou_targets_ini = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_target.detach(),
                    is_aligned=True).clamp(min=1e-6).view(-1)
                pos_ious = iou_targets_ini.clone().detach()

                def normalize(x):
                    EPS = 1e-6
                    t1 = x.min()
                    t2 = min(1., x.max())
                    y = (x - t1 + EPS) / (t2 - t1 + EPS)
                    return y
                joint_weight = (
                    pos_ious * normalize(bbox_targets[pos_inds, -1])).pow(0.5)

            loss_cls_ = self.loss_cls(
                cls_score,
                labels,
                joint_weight,
                weight=weight,
                avg_factor=avg_factor,
                reduction_override=reduction_override)

            if isinstance(loss_cls_, dict):
                losses.update(loss_cls_)
            else:
                losses['loss_cls'] = loss_cls_
            losses['acc'] = accuracy(cls_score, labels)
            fg_mask = labels != self.num_classes
            losses['fg_acc'] = accuracy(
                cls_score[fg_mask], labels[fg_mask])
        else:
            losses['loss_cls'] = cls_score.sum() * 0
        losses['loss_bbox'] = losses_bbox
        losses['losses_iou'] = losses_iou

        if contrast_feature is not None:
            if self.loss_contrast.stage == 1:
                self.loss_contrast.init_proxies(
                    contrast_feature, labels, cls_score)
            if self.loss_contrast.stage == 2:
                loss_contrast = self.loss_contrast(
                    contrast_feature, labels)
                losses['loss_contrast'] = loss_contrast

        return losses

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        roi = roi[:, :-1]
        return super()._predict_by_feat_single(
            roi,
            cls_score,
            bbox_pred,
            img_meta,
            rescale,
            rcnn_test_cfg)


@MODELS.register_module()
class Shared2FCBBoxARLContrastHead(ConvFCBBoxARLContrastHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxARLContrastHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@MODELS.register_module()
class Shared4Conv1FCBBoxARLContrastHead(ConvFCBBoxARLContrastHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxARLContrastHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@MODELS.register_module()
class Decoupled2FC2Conv1FCBBoxARLContrastHead(ConvFCBBoxARLContrastHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Decoupled2FC2Conv1FCBBoxARLContrastHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=2,
            num_cls_fcs=1,
            num_reg_convs=0,
            num_reg_fcs=2,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
