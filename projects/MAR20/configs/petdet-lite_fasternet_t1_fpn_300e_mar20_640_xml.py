_base_ = [
    './petdet-lite_r50_fpn_300e_mar20_640.py',
]

# optimizer
model = dict(
    backbone=dict(
        type='fasternet_t1',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/lwt/work/mmdetection/projects/MAR20/model_ckpt/fasternet_t1-epoch.291-val_acc1.76.2180.pth',
        ),
    ),
    neck=dict(in_channels=[64, 128, 256, 512], out_channels=128),
    fusion=dict(feat_channels=128),
    rpn_head=dict(in_channels=128, feat_channels=128),
    roi_head=dict(
        bbox_roi_extractor=dict(
            out_channels=128,
        ),
        bbox_head=dict(
            in_channels=128,
            fc_out_channels=768,
            conv_out_channels=128,
            contrast_out_channels=384,
            loss_contrast=dict(
                size_contrast=384
            )
        )
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.1)
    )
)

# optimizer
optim_wrapper = dict(optimizer=dict(weight_decay=0.05),
                     clip_grad=dict(max_norm=35, norm_type=2))

# test_dataloader = dict(
#     dataset=dict(
#         indices=10
#     )
# )
test_evaluator = dict(
    type='FHBMetric',
    format_only=True,
    outfile_prefix='./test_output',
)
