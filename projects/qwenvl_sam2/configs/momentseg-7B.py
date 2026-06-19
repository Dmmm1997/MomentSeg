from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer
from projects.qwenvl_sam2.models.qwenvl import Qwen2_5_VL

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.qwenvl_sam2.models import SAM2TrainRunner, VideoLLaVASAMModel_Qwen
from projects.qwenvl_sam2.datasets import VideoReVOSDataset, VideoMeVISDataset, VideoRefYoutubeVOSDataset, video_lisa_collate_fn, VideoSAM2Dataset, ReasonSegDataset
from projects.qwenvl_sam2.datasets import CharadesDataset, ActivityNetDataset
from projects.qwenvl_sam2.datasets import VideoChatUniViDataset
from projects.qwenvl_sam2.datasets import RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset
from projects.qwenvl_sam2.datasets import LLaVADataset
from projects.qwenvl_sam2.datasets import ReferSegmDataset
from projects.qwenvl_sam2.models.preprocess.image_resize import DirectResize
from projects.qwenvl_sam2.models.preprocess.qwen_preprocess import QwenPrepocessor

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = './pretrained/Qwen2.5-VL-7B-Instruct'
pretrained_pth = None

# Data
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192
img_min_pixels = 4*4*28*28        # image referring task
img_max_pixels = 20*20*28*28 
min_pixels = 4*4*28*28            # video referring task
max_pixels = 16*16*28*28
video_min_pixels = 1*1*28*28        # video referring task
video_max_pixels = 4*4*28*28
VTG_video_min_pixels = 4*4*28*28    # video temporal grounding task
VTG_video_max_pixels = 8*8*28*28
single_image_mode=True # False = AnyRES
sampled_frames=5
video_max_frames=50
VTG_video_max_frames=100
find_key_frames=True
exist_frame_token=False
find_sft=False
perframe_seg=False
video_mode="qwen_video"

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 10000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['[SEG]', '[FIND]', '<p>', '</p>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

video_processor = dict(
    type=QwenPrepocessor,
    model_path=path,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    video_min_pixels=video_min_pixels,
    video_max_pixels=video_max_pixels,
)

image_processor = dict(
    type=QwenPrepocessor,
    model_path=path,
    min_pixels=img_min_pixels,
    max_pixels=img_max_pixels,
    video_min_pixels=video_min_pixels,
    video_max_pixels=video_max_pixels,
)

VTG_video_processor = dict(
    type=QwenPrepocessor,
    model_path=path,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    video_min_pixels=VTG_video_min_pixels,
    video_max_pixels=VTG_video_max_pixels,
)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel_Qwen,
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=Qwen2_5_VL,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='FEATURE_EXTRACTION'),
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
        # cfg_path="configs/sam2.1/sam2.1_hiera_l.yaml",
        # ckpt_path="sam2.1_hiera_large.pt",
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    # loss_sample_points=False,
    bs=batch_size,
    arch_type="qwen",
    find_key_frames=find_key_frames,
    perframe_seg=perframe_seg,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = './data/'
VIDEO_DATA_ROOT = DATA_ROOT + 'video_datas/'

############### video res
data_root_revos = VIDEO_DATA_ROOT + 'revos/'
video_revos_image_folder = data_root_revos
video_revos_expression_file = data_root_revos + 'meta_expressions_train_.json'
video_revos_mask_file = data_root_revos + 'mask_dict.json'

data_root_mevis = VIDEO_DATA_ROOT + 'mevis/train/'
video_mevis_image_folder = data_root_mevis + 'JPEGImages'
video_mevis_expression_file = data_root_mevis + 'meta_expressions.json'
video_mevis_mask_file = data_root_mevis + 'mask_dict.json'

data_root_refytvos = VIDEO_DATA_ROOT + 'rvos/'
video_refytvos_image_folder = data_root_refytvos + 'train/JPEGImages/'
video_refytvos_expression_file = data_root_refytvos + 'meta_expressions/train/meta_expressions.json'
video_refytvos_mask_file = data_root_refytvos + 'mask_dict.pkl'

video_revos_dataset = dict(
    type=VideoReVOSDataset,
    image_folder=video_revos_image_folder,
    expression_file=video_revos_expression_file,
    mask_file=video_revos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=20,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=sampled_frames,
    arch_type=video_mode,
    preprocessor=video_processor,
    video_max_frames=video_max_frames,
    find_key_frames=exist_frame_token
)

video_mevis_dataset = dict(
    type=VideoMeVISDataset,
    image_folder=video_mevis_image_folder,
    expression_file=video_mevis_expression_file,
    mask_file=video_mevis_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=10,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=sampled_frames,
    arch_type=video_mode,
    preprocessor=video_processor,
    video_max_frames=video_max_frames,
    find_key_frames=exist_frame_token,
)

video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset,
    image_folder=video_refytvos_image_folder,
    expression_file=video_refytvos_expression_file,
    mask_file=video_refytvos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=10,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=sampled_frames,
    arch_type=video_mode,
    preprocessor=video_processor,
    video_max_frames=video_max_frames,
    find_key_frames=exist_frame_token,
)

################### Video chat
data_root_video_chatunivi = VIDEO_DATA_ROOT + 'chat_univi/'
video_chatunivi_image_folder = data_root_video_chatunivi + 'Activity_Videos/'
video_chatunivi_json_file = data_root_video_chatunivi+ 'video_chat.json'

video_qa_dataset = dict(
    type=VideoChatUniViDataset,
    image_folder=video_chatunivi_image_folder,
    json_file=video_chatunivi_json_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=sampled_frames,
    arch_type="qwen",
    preprocessor=video_processor,
)


################## video temporal grounding
data_root_TVG = DATA_ROOT + 'VTG/NumPro_FT/'
TVG_image_folder = data_root_TVG + 'videos_1FPS/'
charades_expression_file = data_root_TVG+ 'train.caption_coco_format.json'
video_charades_dataset = dict(
    type=CharadesDataset,
    image_folder=TVG_image_folder,
    expression_file=charades_expression_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    repeats=10,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=sampled_frames,
    arch_type=video_mode,
    preprocessor=VTG_video_processor,
    video_max_frames=VTG_video_max_frames,
    find_key_frames=find_key_frames,
    find_sft=find_sft
)

activitynet_expression_file = data_root_TVG+ 'activitynet_captions_train.json'
video_activitynet_dataset = dict(
    type=ActivityNetDataset,
    image_folder=TVG_image_folder,
    expression_file=activitynet_expression_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    repeats=5,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=sampled_frames,
    arch_type=video_mode,
    preprocessor=VTG_video_processor,
    video_max_frames=VTG_video_max_frames,
    find_key_frames=find_key_frames,
    find_sft=find_sft
)

################## image chat
LLAVA_ROOT = DATA_ROOT + 'llava_data/'
llava_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path=LLAVA_ROOT + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder=LLAVA_ROOT + 'llava_images/',
    arch_type="qwen",
    preprocessor=image_processor,
)

################## image res
RES_ROOT = DATA_ROOT + 'ref_seg/'
refcoco_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=RES_ROOT + 'refcoco',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    repeat=2,
    arch_type="qwen",
    preprocessor=image_processor,
    single_image_mode=single_image_mode,
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=RES_ROOT + 'refcoco+',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    repeat=2,
    arch_type="qwen",
    preprocessor=image_processor,
    single_image_mode=single_image_mode,
)
refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root= RES_ROOT + 'refcocog',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    repeat=2,
    arch_type="qwen",
    preprocessor=image_processor,
    single_image_mode=single_image_mode,
)

#############################reasonseg
REASONSEG_ROOT = DATA_ROOT + 'reason_seg/'
reasonseg_dataset=dict(
    type=ReasonSegDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root= REASONSEG_ROOT + 'train',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    repeat=10,
    arch_type="qwen",
    preprocessor=image_processor,
    single_image_mode=single_image_mode,
)


# image gcg datas
glamm_data_root = DATA_ROOT + 'glamm_data/'

refcocog_image_path = glamm_data_root + 'images/coco2014/train2014/'
refcocog_ann_file = glamm_data_root + 'annotations/RefCOCOg_GCG_train.json'

grandf_image_path = glamm_data_root + 'images/grandf/train/'
grandf_ann_file = glamm_data_root + 'annotations/GranDf_HA_GCG_train.json'

flickr_image_path = glamm_data_root + 'images/flickr30k/Flickr30K/'
flickr_ann_file = glamm_data_root + 'annotations/flickr_mergedGT_GCG_train.json'

psg_image_path = glamm_data_root + 'images/coco2017/'
psg_ann_file = glamm_data_root + 'annotations/OpenPsgGCG_train.json'

glamm_refcocog_dataset = dict(
    type=RefCOCOgGCGDataset,
    image_folder=refcocog_image_path,
    data_path=refcocog_ann_file,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    arch_type="qwen",
    preprocessor=image_processor,
)

glamm_grandf_dataset = dict(
    type=GranDfGCGDataset,
    data_path=grandf_ann_file,
    image_folder=grandf_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=10,
    arch_type="qwen",
    preprocessor=image_processor,
)

glamm_psg_dataset = dict(
    type=OpenPsgGCGDataset,
    data_path=psg_ann_file,
    image_folder=psg_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    arch_type="qwen",
    preprocessor=image_processor,
)

glamm_flickr_dataset = dict(
    type=FlickrGCGDataset,
    data_path=flickr_ann_file,
    image_folder=flickr_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    arch_type="qwen",
    preprocessor=image_processor,
)

# sam2 data
data_sam2_folder = VIDEO_DATA_ROOT + 'sam_v_full/'
data_sam2_expression_file = VIDEO_DATA_ROOT + 'sam_v_final_custom.json'

video_sam2_dataset = dict(
    type=VideoSAM2Dataset,
    sam2_folder=data_sam2_folder,
    expression_file=data_sam2_expression_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=sampled_frames,
    select_number=5,
    preprocessor=video_processor,
    arch_type="qwen",
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # ref seg
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # reason seg
        reasonseg_dataset,
        # image qa
        # llava_vqa_dataset,
        # video res
        video_mevis_dataset, video_refytvos_dataset,
        # video reason seg
        video_revos_dataset,
        # video chat
        video_qa_dataset,
        # sam2
        video_sam2_dataset,
        # gcg data
        # glamm_psg_dataset,
        # glamm_grandf_dataset,
        # glamm_flickr_dataset,
        # glamm_refcocog_dataset,
        # temporal video grounding
        video_charades_dataset, video_activitynet_dataset,
    ]
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    # sampler=dict(
    #     type='DefaultSampler',
    #     shuffle=True),
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
