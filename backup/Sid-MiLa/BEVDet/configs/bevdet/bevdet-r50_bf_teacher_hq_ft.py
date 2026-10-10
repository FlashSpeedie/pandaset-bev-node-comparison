_base_ = ['bevdet-r50.py']
data_config = dict(
    cams=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',
          'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'],
    Ncams=6,
    input_size=(256, 704),
    src_size=(1200, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
class_names = [
    'car','truck','construction_vehicle','bus','trailer','barrier',
    'motorcycle','bicycle','pedestrian','traffic_cone']
train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs','gt_bboxes_3d','gt_labels_3d']),
]
test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333,800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['img_inputs'])]),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='NuScenesDataset',
        data_root='./',
        ann_file='custom_nuscenes_infos_bf_teacher_hq.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        test_mode=False,
        use_valid_flag=True,
        img_info_prototype='bevdet',
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(
        type='NuScenesDataset',
        data_root='./',
        ann_file='custom_nuscenes_infos_bf_teacher_hq.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        img_info_prototype='bevdet',
        filter_empty_gt=False),
    test=dict(
        type='NuScenesDataset',
        data_root='./',
        ann_file='custom_nuscenes_infos_clean.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        img_info_prototype='bevdet'),
)
optimizer = dict(type='AdamW', lr=5e-6, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60, 85])
total_epochs = 100
runner = dict(type='EpochBasedRunner', max_epochs=100)
log_config = dict(interval=20)
# save every 10 epochs so you can test 50/60/70/80/90/100 later
checkpoint_config = dict(interval=10, max_keep_ckpts=20)
evaluation = dict(interval=100)
load_from = 'work_dirs/bevdet_bf_teacher_ft/epoch_8_ema.pth'
