_base_ = [
    '../../MAR20/configs/faster-rcnn_r50_fpn_3x_mar20.py',
]

model = dict(
    neck=dict(
        start_level=1,
        add_extra_convs='on_output',  # use P5
        relu_before_extra_convs=True),
    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='TOODHead',
        # num_classes = 1 for rpn,
        # if num_classes > 1, it will be set to 1 in
        # TwoStageDetector automatically
        num_classes=1,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=dict(  # update featmap_strides
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128])),
    train_cfg=dict(
        rpn=dict(
            _delete_=True,
            initial_epoch=10000,
            initial_assigner=dict(type='ATSSAssigner', topk=9),
            assigner=dict(type='TaskAlignedAssigner', topk=13),
            alpha=1,
            beta=6,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            score_thr=0.0,
            nms=None,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            _delete_=True,
            nms_pre=2000,
            max_per_img=1000,
            score_thr=0.0,
            nms=None,
            min_bbox_size=0)
    )
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),  # Slowly increase lr, otherwise loss becomes NAN
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

log_processor = dict(
    custom_cfg=[
        dict(data_src='acc',
             method_name='mean',
             window_size=50),
        dict(data_src='fg_acc',
             method_name='mean',
             window_size=50),
    ])
