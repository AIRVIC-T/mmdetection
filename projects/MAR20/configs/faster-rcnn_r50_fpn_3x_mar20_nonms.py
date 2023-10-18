_base_ = [
    './faster-rcnn_r50_fpn_3x_mar20.py'
]

# model settings
model = dict(
    train_cfg=dict(
        rpn_proposal=dict(
            nms=None
        )),
    test_cfg=dict(
        rpn=dict(
            nms=None
        )

    )
)
