_base_ = [
    '../_base_/datasets/iBioHash.py',
    '../_base_/schedules/imagenet_bs2048_AdamW.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        frozen_stages=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint="https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth",
            prefix='backbone')
        ),
    neck=None,
    head=dict(
        type='GreedyHashHead',
        bit=48,
        num_classes=1000,
        alpha=0.01,
        in_channels=768,
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
    ))
# model_wrapper_cfg=dict(type='MMFullyShardedDataParallel', cpu_offload=False)

#dataset settings
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
)
gpus = 2

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.3),
    clip_grad=dict(max_norm=1.0),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)

# learning policy
warmup_epochs = 5  # about 10000 iterations for ImageNet-1k
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=warmup_epochs)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200)
val_cfg = None
test_cfg = None

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=gpus * train_dataloader['batch_size'])


#runtime
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=200),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=20),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)




