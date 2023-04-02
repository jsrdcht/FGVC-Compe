_base_ = [
    '../_base_/datasets/iBioHash.py',
    '../_base_/schedules/imagenet_bs2048_AdamW.py',
    '../_base_/default_runtime.py'
]


model = dict(
    type='ImageClassifier',
    # backbone=dict(
    #     type='VisionTransformer',
    #     arch='b',
    #     img_size=224,
    #     patch_size=16,
    #     drop_rate=0.1,
    #     frozen_stages=-1,
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint="https://download.openmmlab.com/mmclassification/v0/beit/beitv2-base_3rdparty_in1k_20221114-73e11905.pth",
    #         prefix='backbone')
    #     ),
    backbone=dict(
        type='BEiT',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        avg_token=True,
        output_cls_token=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint="https://download.openmmlab.com/mmclassification/v0/beit/beitv2-base_3rdparty_in1k_20221114-73e11905.pth",
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
    batch_size=32,
    num_workers=4,
)
gpus = 2

layer_decay = 0.8
lr = 1e-4
# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.3),
    clip_grad=dict(max_norm=1.0),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0, lr_mult=pow(layer_decay, 13)),
        '.pos_embed': dict(decay_mult=0.0, lr_mult=pow(layer_decay, 13)),

        'backbone.patch_embed': dict(lr_mult=pow(layer_decay, 13)),
        'backbone.layer0': dict(lr_mult=pow(layer_decay, 12)),
        'backbone.layer1': dict(lr_mult=pow(layer_decay, 11)),
        'backbone.layer2': dict(lr_mult=pow(layer_decay, 10)),
        'backbone.layer3': dict(lr_mult=pow(layer_decay, 9)),
        'backbone.layer4': dict(lr_mult=pow(layer_decay, 8)),
        'backbone.layer5': dict(lr_mult=pow(layer_decay, 7)),
        'backbone.layer6': dict(lr_mult=pow(layer_decay, 6)),
        'backbone.layer7': dict(lr_mult=pow(layer_decay, 5)),
        'backbone.layer8': dict(lr_mult=pow(layer_decay, 4)),
        'backbone.layer9': dict(lr_mult=pow(layer_decay, 3)),
        'backbone.layer10': dict(lr_mult=pow(layer_decay, 2)),
        'backbone.layer11': dict(lr_mult=pow(layer_decay, 1)),
        
    }),
)

# learning policy
warmup_epochs = 3  # about 10000 iterations for ImageNet-1k
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
train_cfg = dict(by_epoch=True, max_epochs=30)
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
    checkpoint=dict(type='CheckpointHook', interval=10),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)




