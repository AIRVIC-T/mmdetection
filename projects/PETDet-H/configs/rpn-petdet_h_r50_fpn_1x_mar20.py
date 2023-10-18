_base_ = [
    './rpn-rcnn-bcfn_r50_fpn_3x_mar20.py'
]

model = dict(
    rpn_head=dict(
        out_score=True
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractorWithScore',
        ),
        bbox_head=dict(
            type='Shared2FCBBoxARLHead',
            loss_cls=dict(
                type='AdaptiveRecognitionLoss',
                beta=3.0,
                gamma=1.5),
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            # nms=None,
        ),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='PseudoSampler')),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            # nms=None
        ),
        rcnn=dict(
            nms_pre=1000,
            max_per_img=100)
    )
)

log_processor = dict(
    custom_cfg=[
        dict(data_src='acc',
             method_name='mean',
             window_size=50),
        dict(data_src='fg_acc',
             method_name='mean',
             window_size=50),
    ])
