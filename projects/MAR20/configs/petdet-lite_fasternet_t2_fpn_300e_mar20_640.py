_base_ = [
    './petdet-lite_r50_fpn_300e_mar20_640.py',
]

# optimizer
model = dict(
    backbone=dict(
        type='fasternet_t2',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/lwt/work/mmdetection/projects/MAR20/model_ckpt/fasternet_t2-epoch.289-val_acc1.78.8860.pth',
        ),
    ),
    neck=dict(in_channels=[96, 192, 384, 768], out_channels=192),
    fusion=dict(feat_channels=192),
    rpn_head=dict(in_channels=192, feat_channels=192),
    roi_head=dict(
        bbox_roi_extractor=dict(
            out_channels=192,
        ),
        bbox_head=dict(
            in_channels=192,
            fc_out_channels=1024,
            conv_out_channels=192,
            contrast_out_channels=384,
            loss_contrast=dict(
                size_contrast=384
            )
        )
    )
)

# optimizer
optim_wrapper = dict(optimizer=dict(weight_decay=0.05),
                     clip_grad=dict(max_norm=35, norm_type=2))
