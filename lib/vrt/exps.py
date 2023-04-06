"""

The experimental meshgrids used
for training and testing in our project.

"""

# -- cache_io for meshgrid --
import cache_io

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- local --
from .common import dcat

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         Train/Test the "init" Values
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def exp_init(iexps = None, mode = "train"):
    if mode == "train":
        return exp_train_init(iexps)
    elif mode == "test":
        return exp_test_init(iexps)
    else:
        raise ValueError(f"Uknown mode [{mode}]")

def exp_train_init(iexps = None):
    isize = ["256_256"]
    expl = {"isize":isize}
    dcat(expl,iexps)
    expl = exp_default_init(expl)
    return expl

def exp_test_init(iexps = None):
    chkpt = [""]
    use_train = ['false']
    expl = {"chkpt":chkpt,"use_train":use_train}

    dcat(expl,iexps)
    expl = exp_default_init(expl)
    return expl

def exp_default_init(iexps = None):
    # -- input checking --
    if iexps is None: iexps = {}
    assert isinstance(iexps,dict),"Must be dict"

    # -- defaults --
    k,ws,ps,wt = [-1],[8],[1],[0]
    filter_by_attn_pre = ["false"]
    filter_by_attn_post = ["false"]
    attn_mode = ["product_stnls"]
    pt,stride0,stride1 = [1],[1],[1]
    dil,nbwd = [1],[1]
    rbwd,exact = ["true"],["false"]
    bs,flow = [-1],['false']
    load_pretrained = ["true"]
    freeze = ["false"]
    pretrained_path = [""]
    pretrained_prefix = ["module."]
    in_attn_mode = ["window_default"] # original attn mode
    embed_dim = [32]
    attn_reset = ['f-f-f-f-f']
    freeze = ['f-f-f-f-f']

    # -- grid --
    exp_lists = {"attn_mode":attn_mode,"ws":ws,"wt":wt,"k":k,"ps":ps,
                 "pt":pt,"stride0":stride0,"stride1":stride1,"dil":dil,
                 "nbwd":nbwd,"rbwd":rbwd,"exact":exact,"bs":bs,'flow':flow,
                 "load_pretrained":load_pretrained,"pretrained_path":pretrained_path,
                 "pretrained_prefix":pretrained_prefix,"in_attn_mode":in_attn_mode,
                 "freeze":freeze,"filter_by_attn_pre":filter_by_attn_pre,
                 "filter_by_attn_post":filter_by_attn_post,"embed_dim":embed_dim,
                 "attn_reset":attn_reset,"in_attn_mode":in_attn_mode,
                 "attn_mode":attn_mode,"attn_reset":attn_reset,"freeze":freeze}
    # -- apped new values --
    dcat(exp_lists,iexps) # input overwrites defaults
    return exp_lists

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         >>> Empty Formatting Here <<<
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#     Run Denoising Experiments
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def exps_rgb_denoising(iexps=None,mode="train"):
    if mode == "train":
        return exps_rgb_denoising_train(iexps)
    elif mode == "test":
        return exps_rgb_denoising_test(iexps)
    else:
        raise ValueError("Unable to verify new code.")

def exps_rgb_denoising_train(iexps=None):

    # -- init --
    expl = exp_init(iexps,"train")
    expl['freeze'] = ['false']

    # -- [exp a] step 0 --
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["w-w-w-w-w"]
    expl['attn_reset'] = ["t-t-f-f-f"]
    exps = cache_io.mesh_pydicts(expl) # create mesh

    return exps

def exps_rgb_denoising_test(iexps=None):

    # -- init --
    expl = exp_init(iexps,"test")
    expl['use_train'] = ['false']
    expl['attn_reset'] = ["f-f-f-f-f"]
    expl['pretrained_prefix'] = ["net."]

    # -- [exp a] step 0 --
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["w-w-w-w-w"]
    expl['attn_reset'] = ["t-t-f-f-f"]
    exps = cache_io.mesh_pydicts(expl) # create mesh

    return exps



