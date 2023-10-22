_base_ = [
    './petdet_r50_fpn_3x_mar20.py'
]

load_from = '/home/lwt/work/PETDet/work_dirs/petdet_r50_fpn_1x_fair1m_le90/latest.pth'

model = dict(
    neck=dict(
        add_extra_convs='on_input',
    )
)
