
# -- python imports --
import os
import warnings
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
from torch.nn.modules.utils import _pair, _single
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from einops.layers.torch import Rearrange

# -- project imports --
import torch as th
import dnls

# -- local imports --
from . import aligned
from .aligned import flow_warp
# from .flows import get_warped_frames
# # get_warp_2f
from .tmsa import TMSAG
from .mlp import Mlp_GEGLU
from .deform_conv import DCNv2PackFlowGuided

class Stage(nn.Module):
    """Residual Temporal Mutual Self Attention Group and Parallel Warping.

    Args:
        in_dim (int): Number of input channels.
        dim (int): Number of channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        reshape (str): Downscale (down), upscale (up) or keep the size (none).
        max_residue_magnitude (float): Maximum magnitude of the residual of optical flow.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 in_dim,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mul_attn_ratio=0.75,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 pa_frames=2,
                 deformable_groups=16,
                 reshape=None,
                 max_residue_magnitude=10,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 warp_mode="default"
                 ):
        super(Stage, self).__init__()
        self.pa_frames = pa_frames
        self.warp_mode =  warp_mode

        # reshape the tensor
        if reshape == 'none':
            self.reshape = nn.Sequential(Rearrange('n c d h w -> n d h w c'),
                                         nn.LayerNorm(dim),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'down':
            self.reshape = nn.Sequential(Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
                                         nn.LayerNorm(4 * in_dim), nn.Linear(4 * in_dim, dim),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'up':
            self.reshape = nn.Sequential(Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2),
                                         nn.LayerNorm(in_dim // 4), nn.Linear(in_dim // 4, dim),
                                         Rearrange('n d h w c -> n c d h w'))

        # mutual and self attention
        self.residual_group1 = TMSAG(dim=dim,
                                     input_resolution=input_resolution,
                                     depth=int(depth * mul_attn_ratio),
                                     num_heads=num_heads,
                                     window_size=(2, window_size[1], window_size[2]),
                                     mut_attn=True,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     use_checkpoint_attn=use_checkpoint_attn,
                                     use_checkpoint_ffn=use_checkpoint_ffn
                                     )
        self.linear1 = nn.Linear(dim, dim)

        # only self attention
        self.residual_group2 = TMSAG(dim=dim,
                                     input_resolution=input_resolution,
                                     depth=depth - int(depth * mul_attn_ratio),
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mut_attn=False,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     use_checkpoint_attn=True,
                                     use_checkpoint_ffn=use_checkpoint_ffn
                                     )
        self.linear2 = nn.Linear(dim, dim)

        # parallel warping
        if self.pa_frames:
            self.pa_deform = DCNv2PackFlowGuided(dim, dim, 3, padding=1,
                                deformable_groups=deformable_groups,
                                max_residue_magnitude=max_residue_magnitude,
                                pa_frames=pa_frames)
            self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def forward(self, x, flows_backward, flows_forward):
        # print("stage input: ",x.shape)
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

        if self.pa_frames:
            x = x.transpose(1, 2)
            x_backward, x_forward = get_aligned(x,flows_backward,flows_forward,
                                                self.pa_frames,self.pa_deform,
                                                self.warp_mode)
            x = self.pa_fuse(
                torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)
            ).permute(0, 4, 1, 2, 3)
        return x


def get_aligned(vid,bflow,fflow,pa_frames,pa_deform,mode):
    if mode == "default":
        fxn_name = f'get_aligned_feature_{pa_frames}frames'
        aligned_fxn = getattr(aligned,fxn_name)
        bwd,fwd = aligned_fxn(vid, bflow, fflow, pa_deform)
    else:
        # print(vid.shape)
        fwd, bwd = dnls.nn.flow_patches_f.get_warp_2f(vid,fflow[0],bflow[0],
                                                      k=pa_frames-1,ws=5,warp_ps=4,
                                                      ps=5,stride0=2)
        dims = th.arange(fwd.ndim)[2:].numpy().tolist()
        # print("fwd: ",th.mean((fwd - vid)**2,dim=dims))
        # print("bwd: ",th.mean((bwd - vid)**2,dim=dims))
        # print("delta: ",th.mean((fwd - bwd)**2,dim=dims))
        # print("fwd.shape,pa_frames: ",fwd.shape,pa_frames)
        # fwd,bwd = get_warp_2f(vid,fflow[0],bflow[0],k=self.pa_frames)
        fwd,bwd = apply_pa_deform(vid,fwd,bwd,fflow,bflow,pa_deform)
        # bwd,fwd = aligned.get_aligned_feature_2frames(vid, bflow, fflow, pa_deform)
    return bwd,fwd

    # n = x.size(1)
    # x_backward = [torch.zeros_like(x[:, -1, ...])]
    # for i in range(n - 1, 0, -1):
    #     x_i = x[:, i, ...]
    #     flow = flows_backward[0][:, i - 1, ...]
    #     x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
    #     x_backward.insert(0, pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

    # # forward
    # x_forward = [torch.zeros_like(x[:, 0, ...])]
    # for i in range(0, n - 1):
    #     x_i = x[:, i, ...]
    #     flow = flows_forward[0][:, i, ...]
    #     x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
    #     x_forward.append(pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

    # return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

def apply_pa_deform(vid,fwd,bwd,fflow,bflow,pa_deform):

    T = vid.shape[1]

    deform_fwd = th.zeros_like(vid)
    for i in range(T-1):
        x_i = vid[:, i, ...]
        flow = fflow[0][:, i, ...]
        x_i_warped = fwd[:, i]
        # x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
        # print(i,th.mean((x_i-x_i_warped)**2).item())
        deform_fwd[:,i+1] = pa_deform(x_i, [x_i_warped], vid[:, i + 1, ...], [flow])

    deform_bwd = th.zeros_like(vid)
    for i in range(T - 1, 0, -1):
        x_i = vid[:, i, ...]
        flow = bflow[0][:, i - 1, ...]
        x_i_warped = bwd[:, i]
        # x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
        # print(i,th.mean((x_i-x_i_warped)**2).item())
        # print(i,x_i.shape,x_i_warped.shape)
        deform_bwd[:,i-1] = pa_deform(x_i, [x_i_warped], vid[:, i - 1, ...], [flow])


    return deform_fwd,deform_bwd
