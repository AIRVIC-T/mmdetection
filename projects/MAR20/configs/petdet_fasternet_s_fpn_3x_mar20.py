_base_ = [
    './petdet_r50_fpn_3x_mar20.py',
]

# optimizer
model = dict(
    backbone=dict(
        type='fasternet_s',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/lwt/work/mmdetection/projects/MAR20/model_ckpt/fasternet_s-epoch.299-val_acc1.81.2840.pth',
            ),
        ),
    neck=dict(in_channels=[128, 256, 512, 1024]))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)