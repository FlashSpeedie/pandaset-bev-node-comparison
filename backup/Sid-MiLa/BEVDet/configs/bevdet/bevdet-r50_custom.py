data_config = dict(
    cams=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
          'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
    Ncams=6,
    input_size=(256, 704),
    src_size=(1080, 1920),
    resize=(-0.06, 0.19),
    rot=(-5.4, 5.4),
    crop=(0, 0),
    crop_h=(0.0, 0.0),
    crop_w=(0.0, 0.0),
    flip=True,
    bwd_ext=False)

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flat_translation=[-0.0, -0.0, -0.0],
    flip_dx=True,
    flip_dy=True)

class_names = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
]

_base_ = ['bevdet-r50.py']

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ]
    )
]

data = dict(
    test=dict(
        pipeline=test_pipeline
    )
)
