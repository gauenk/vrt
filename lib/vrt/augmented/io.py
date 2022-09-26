
# -- misc --
import sys,os,copy
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .vrt import VRT as net

# -- misc imports --
from ..common import optional as _optional
from ..utils.model_utils import load_checkpoint_module,load_checkpoint_qkv
# from ..utils.model_utils import load_checkpoint_mix_qkv
from ..utils.model_utils import remove_lightning_load_state
from ..utils.model_utils import filter_rel_pos,get_recent_filename

# -- auto populate fields to extract config --
_fields = []
def optional_full(init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

# -- load model --
def load_model(*args,**kwargs):

    # -- allows for all keys to be aggregated at init --
    init = _optional(kwargs,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # -- defaults changed by noise version --
    noise_version = optional(kwargs,'noise_version',"noise")
    if "noise" in noise_version:
        default_modulator = True
        default_depth = [1, 2, 8, 8, 2, 8, 8, 2, 1]
        # default_modulator = False
        # default_depth = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    elif noise_version == "blur":
        default_modulator = True
        default_depth = [1, 2, 8, 8, 2, 8, 8, 2, 1]
    else:
        raise ValueError(f"Uknown noise version [{noise_version}]")

    # -- get cfg --
    nchnls = optional(kwargs,'nchnls',3)
    input_size = optional(kwargs,'input_size',128)
    depths = optional(kwargs,'input_depth',default_depth)
    device = optional(kwargs,'device','cuda:0')
    attn_mode = optional(kwargs,'attn_mode','default')
    task = optional(kwargs,'task','denoising')

    # -- break here if init --
    if init: return
    model,datasets,args = init_from_task(task)
    return model

# -- run to populate "_fields" --
load_model(__init=True)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Extracting Relevant Fields from Larger Dict
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_model_io(cfg):
    # -- used to be manual --
    # fields = ["attn_mode","ws","wt","k","ps","pt","stride0",
    #           "stride1","dil","nbwd","rbwd","exact","bs",
    #           "noise_version","filter_by_attn"]

    # -- auto populated fields --
    fields = _fields
    model_cfg = {}
    for field in fields:
        if field in cfg:
            model_cfg[field] = cfg[field]
    return model_cfg

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   Use the original io values from the original net.
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def init_from_task(task,**kwargs):
    ''' prepare model and dataset according to args.task. '''

    # define model
    args = edict()
    if task == '001_VRT_videosr_bi_REDS_6frames':
        model = net(upscale=4, img_size=[6,64,64], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif task == '002_VRT_videosr_bi_REDS_16frames':
        model = net(upscale=4, img_size=[16,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=6, deformable_groups=24)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [8,8,8]
        args.nonblind_denoising = False

    elif task in ['003_VRT_videosr_bi_Vimeo_7frames', '004_VRT_videosr_bd_Vimeo_7frames']:
        model = net(upscale=4, img_size=[8,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=4, deformable_groups=16)
        datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 4
        args.window_size = [8,8,8]
        args.nonblind_denoising = False

    elif task in ["deblur_dvd",'005_VRT_videodeblurring_DVD']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['DVD10']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif task in ["deblur_gopro",'006_VRT_videodeblurring_GoPro']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['GoPro11-part1', 'GoPro11-part2']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif task in ["deblur_reds",'007_VRT_videodeblurring_REDS']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['REDS4']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif task in ["denoising","denoise_davis",'008_VRT_videodenoising_DAVIS',"rgb_denoise"]:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8],
                    depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10],
                    embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2,
                    deformable_groups=16,
                    nonblind_denoising=True)
        datasets = ['Set8', 'DAVIS-test']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = True
    elif task == '009_VRT_videofi_Vimeo_4frames':
        model = net(upscale=1, out_chans=3, img_size=[4,192,192], window_size=[4,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=0)
        datasets = ['UCF101', 'DAVIS-train']  # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 1
        args.window_size = [4,8,8]
        args.nonblind_denoising = False
    else:
        raise ValueError(f"Uknown task [{task}]")
    return model,datasets,args

