_base_ = [
    './qpn-rcnn_r50_fpn_3x_mar20.py'
]

model = dict(
    neck=dict(
        start_level=0,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        num_outs=6),
    fusion=dict(
        type='BCFN',
        num_ins=5,
        feat_channels=256,
    ),
    rpn_head=dict(
        start_level=1,
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32]),
    )
)