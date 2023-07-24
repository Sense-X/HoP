_base_ = [
    '../_base_/datasets/nus-mono3d.py', '../_base_/models/fcos3d.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='ViT2d',
        img_size=1280,
        pretrain_img_size=560,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4096 / 1024,
        drop_path_rate=0.4,
        window_size=21,
        window_block_indexes=window_block_indexes,
        residual_block_indexes=residual_block_indexes,
        freeze_patch_embed=True,
        beit_like_gamma=False,
        qkv_bias=True,
        beit_like_qkv_bias=False,
        use_act_checkpoint=True,
        init_cfg=None),
    neck=dict(
        _delete_=True,
        type='SFP2d',
        in_channels=[1024],
        out_channels=256,
        add_extra_convs=True,
        use_act_checkpoint=False,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4))

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
# Optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=24, layer_decay_rate=0.8))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[10,])
total_epochs = 12
evaluation = dict(interval=1)
