
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
# from ..common import optional as _optional
from ..utils.model_utils import load_checkpoint_module,load_checkpoint_qkv
# from ..utils.model_utils import load_checkpoint_mix_qkv
from ..utils.model_utils import remove_lightning_load_state
from ..utils.model_utils import filter_rel_pos,get_recent_filename

# -- io --
# from ..utils import model_io
from dev_basics import arch_io

# -- config --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction

# -- load model --
@econfig.set_init
def load_model(cfg):

    # -- config --
    econfig.init(cfg)
    device = econfig.optional(cfg,"device","cuda:0")
    cfgs = econfig.extract_dict_of_pairs(cfg,{"arch":arch_pairs(),
                                              "io":io_pairs()},restrict=True)
    if econfig.is_init: return

    # -- load model --
    model = net(**cfgs.arch)

    # -- load model --
    load_pretrained(model,cfgs.io)

    # -- to device --
    model = model.to(device)

    return model

def load_pretrained(model,cfg):
    def mod_fxn(state):
        return state['params']
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        arch_io.load_checkpoint(model,cfg.pretrained_path,
                                cfg.pretrained_root,cfg.pretrained_type,
                                mod=mod_fxn)

def io_pairs():
    pretrained_path = "model_zoo/vrt/008_VRT_videodenoising_DAVIS.pth"
    pairs = {"pretrained_load":True,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"mod",
             "pretrained_root":"."}
    return pairs

def arch_pairs():
    pairs = {#"attn_mode":"default",
             #"task":"denoising",
             "upscale":1,
             "img_size":[6,192,192],
             "window_size":[6,8,8],
             "depths":[8,8,8,8,8,8,8,4,4,4,4],
             "indep_reconsts":[9,10],
             "embed_dims":[96,96,96,96,96,96,96,120,120,120,120],
             "num_heads":[6,6,6,6,6,6,6,6,6,6,6],
             "pa_frames":2,
             "deformable_groups":16,
             "nonblind_denoising":True,
             "warp_mode":"default"}
    return pairs

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

