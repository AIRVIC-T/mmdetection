_base_ = [
    './petdet-lite_r50_fpn_200e_dior_640.py',
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
    )
)

# optimizer
optim_wrapper = dict(optimizer=dict(weight_decay=0.001),
                     clip_grad=dict(max_norm=35, norm_type=2))
