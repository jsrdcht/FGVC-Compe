dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=1000,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root='/home/ct/code/fgvc/iBioHash_Train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                scale=224,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.3),
    paramwise_cfg=dict(
        custom_keys=dict(
            {
                '.cls_token': dict(
                    decay_mult=0.0, lr_mult=0.054975581388800036),
                '.pos_embed': dict(
                    decay_mult=0.0, lr_mult=0.054975581388800036),
                'backbone.patch_embed': dict(lr_mult=0.054975581388800036),
                'backbone.layer0': dict(lr_mult=0.06871947673600004),
                'backbone.layer1': dict(lr_mult=0.08589934592000005),
                'backbone.layer2': dict(lr_mult=0.10737418240000006),
                'backbone.layer3': dict(lr_mult=0.13421772800000006),
                'backbone.layer4': dict(lr_mult=0.1677721600000001),
                'backbone.layer5': dict(lr_mult=0.20971520000000007),
                'backbone.layer6': dict(lr_mult=0.2621440000000001),
                'backbone.layer7': dict(lr_mult=0.3276800000000001),
                'backbone.layer8': dict(lr_mult=0.4096000000000001),
                'backbone.layer9': dict(lr_mult=0.5120000000000001),
                'backbone.layer10': dict(lr_mult=0.6400000000000001),
                'backbone.layer11': dict(lr_mult=0.8)
            })),
    clip_grad=dict(max_norm=1.0),
    type='AmpOptimWrapper',
    loss_scale='dynamic')
warmup_epochs = 3
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=3,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-05, by_epoch=True, begin=3)
]
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = None
test_cfg = None
auto_scale_lr = dict(base_batch_size=64)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth',
            prefix='backbone')),
    neck=None,
    head=dict(
        type='GreedyHashHead',
        bit=48,
        num_classes=1000,
        alpha=0.01,
        in_channels=768,
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
gpus = 2
layer_decay = 0.8
lr = 0.0001
launcher = 'pytorch'
work_dir = './results/greedyhash'
seed = '42+deterministic'
