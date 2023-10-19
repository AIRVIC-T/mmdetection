_base_ = [
    './tood-rcnn_r50_fpn_3x_mar20.py'
]
# find_unused_parameters = True

custom_imports = dict(
    imports=['projects.PETDet-H.petdet'], allow_failed_imports=False)

model = dict(
    type='PETDetHorizontal',
    neck=dict(
        start_level=0,
        add_extra_convs='on_output',
        num_outs=6),
    fusion=dict(
        type='BCFN',
        num_ins=5,
        feat_channels=256,
    ),
    rpn_head=dict(
        type='TOODHeadWithScore',
        start_level=1,
        out_score=False
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32]),
    )
)
