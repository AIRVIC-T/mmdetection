_base_ = [
    './petdet_r50_fpn_3x_mar20.py'
]

checkpoint = 'https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlaspretrain/sentinel2/si_sb_resnet50.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', prefix='backbone.resnet.', checkpoint=checkpoint)))

# optim_wrapper = dict(
#     optimizer=dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05),
#     paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
