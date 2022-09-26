
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

# -- local imports --
from .flows import flow_warp


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    aligning features with warps
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_aligned_feature_2frames(x, flows_backward, flows_forward, pa_deform):
    '''Parallel feature warping for 2 frames.'''

    # backward
    n = x.size(1)
    x_backward = [torch.zeros_like(x[:, -1, ...])]
    for i in range(n - 1, 0, -1):
        x_i = x[:, i, ...]
        flow = flows_backward[0][:, i - 1, ...]
        x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
        x_backward.insert(0, pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

    # forward
    x_forward = [torch.zeros_like(x[:, 0, ...])]
    for i in range(0, n - 1):
        x_i = x[:, i, ...]
        flow = flows_forward[0][:, i, ...]
        x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
        x_forward.append(pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

    return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

def get_aligned_feature_4frames(x, flows_backward, flows_forward,pa_deform):
    '''Parallel feature warping for 4 frames.'''

    # backward
    n = x.size(1)
    x_backward = [torch.zeros_like(x[:, -1, ...])]
    for i in range(n, 1, -1):
        x_i = x[:, i - 1, ...]
        flow1 = flows_backward[0][:, i - 2, ...]
        if i == n:
            x_ii = torch.zeros_like(x[:, n - 2, ...])
            flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
        else:
            x_ii = x[:, i, ...]
            flow2 = flows_backward[1][:, i - 2, ...]

        x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
        x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
        x_backward.insert(0,
            pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

    # forward
    x_forward = [torch.zeros_like(x[:, 0, ...])]
    for i in range(-1, n - 2):
        x_i = x[:, i + 1, ...]
        flow1 = flows_forward[0][:, i + 1, ...]
        if i == -1:
            x_ii = torch.zeros_like(x[:, 1, ...])
            flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
        else:
            x_ii = x[:, i, ...]
            flow2 = flows_forward[1][:, i, ...]

        x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
        x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
        x_forward.append(
            pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

    return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

def get_aligned_feature_6frames(x, flows_backward, flows_forward, pa_deform):
    '''Parallel feature warping for 6 frames.'''

    # backward
    n = x.size(1)
    x_backward = [torch.zeros_like(x[:, -1, ...])]
    for i in range(n + 1, 2, -1):
        x_i = x[:, i - 2, ...]
        flow1 = flows_backward[0][:, i - 3, ...]
        if i == n + 1:
            x_ii = torch.zeros_like(x[:, -1, ...])
            flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])
            x_iii = torch.zeros_like(x[:, -1, ...])
            flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
        elif i == n:
            x_ii = x[:, i - 1, ...]
            flow2 = flows_backward[1][:, i - 3, ...]
            x_iii = torch.zeros_like(x[:, -1, ...])
            flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
        else:
            x_ii = x[:, i - 1, ...]
            flow2 = flows_backward[1][:, i - 3, ...]
            x_iii = x[:, i, ...]
            flow3 = flows_backward[2][:, i - 3, ...]

        x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
        x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
        x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i+3 aligned towards i
        x_backward.insert(0,pa_deform(torch.cat([x_i, x_ii, x_iii], 1),
                                    [x_i_warped, x_ii_warped, x_iii_warped],
                                    x[:, i - 3, ...], [flow1, flow2, flow3]))

    # forward
    x_forward = [torch.zeros_like(x[:, 0, ...])]
    for i in range(0, n - 1):
        x_i = x[:, i, ...]
        flow1 = flows_forward[0][:, i, ...]
        if i == 0:
            x_ii = torch.zeros_like(x[:, 0, ...])
            flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
            x_iii = torch.zeros_like(x[:, 0, ...])
            flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
        elif i == 1:
            x_ii = x[:, i - 1, ...]
            flow2 = flows_forward[1][:, i - 1, ...]
            x_iii = torch.zeros_like(x[:, 0, ...])
            flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
        else:
            x_ii = x[:, i - 1, ...]
            flow2 = flows_forward[1][:, i - 1, ...]
            x_iii = x[:, i - 2, ...]
            flow3 = flows_forward[2][:, i - 2, ...]

        x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
        x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
        x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i-3 aligned towards i
        x_forward.append(pa_deform(torch.cat([x_i, x_ii, x_iii], 1),
                                   [x_i_warped, x_ii_warped, x_iii_warped],
                                   x[:, i + 1, ...], [flow1, flow2, flow3]))

    return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]
