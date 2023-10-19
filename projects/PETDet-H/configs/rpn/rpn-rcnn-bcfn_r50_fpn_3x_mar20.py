_base_ = [
    '../../MAR20/configs/faster-rcnn_r50_fpn_3x_mar20.py'
]
#find_unused_parameters = True

custom_imports = dict(
    imports=['projects.PETDet-H.petdet'], allow_failed_imports=False)

model = dict(
    type='PETDetHorizontal',
    neck=dict(
        start_level=0,
        num_outs=5),
    fusion=dict(
        type='BCFN',
        num_ins=5,
        feat_channels=256,
    ),
    rpn_head=dict(
        type='RPNHeadWithScore',
        start_level=0,
        end_level=5,
        out_score=True
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32]),
    )
)
