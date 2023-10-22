# dataset settings
dataset_type = 'XMLDataset'
data_root = 'data/MAR20/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

metainfo = dict(
    classes=(
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
        'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
        'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
        'A20'))

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='ImageSets/Main/train.txt',
        ann_subdir='Annotations/Horizontal Bounding Boxes',
        data_prefix=dict(sub_data_root=''),
        filter_cfg=dict(
            filter_empty_gt=True, min_size=32, bbox_min_size=32),
        test_mode=False,
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='ImageSets/Main/test.txt',
        ann_subdir='Annotations/Horizontal Bounding Boxes',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(
    type='VOCMetric',
    iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    metric='mAP',
    eval_mode='area',
)
test_evaluator = val_evaluator
