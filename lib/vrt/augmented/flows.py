

# -- python imports --
import torch as th
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

# -- proj --
import dnls


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         flow utils
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_flows(x,pa_frames,flow_net):
    ''' Get flows for 2 frames, 4 frames or 6 frames.'''

    if pa_frames == 2:
        flows_backward, flows_forward = get_flow_2frames(x,flow_net)
    elif pa_frames == 4:
        flows_backward_2frames, flows_forward_2frames = get_flow_2frames(x,flow_net)
        flows_backward_4frames, flows_forward_4frames = get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
        flows_backward = flows_backward_2frames + flows_backward_4frames
        flows_forward = flows_forward_2frames + flows_forward_4frames
    elif pa_frames == 6:
        flows_backward_2frames, flows_forward_2frames = get_flow_2frames(x,flow_net)
        flows_backward_4frames, flows_forward_4frames = get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
        flows_backward_6frames, flows_forward_6frames = get_flow_6frames(flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames)
        flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
        flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames

    return flows_backward, flows_forward

def get_flow_2frames(x, flow_net):
    '''Get flow between frames t and t+1 from x.'''

    b, n, c, h, w = x.size()
    x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

    # backward
    flows_backward = flow_net(x_1, x_2)
    flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                      zip(flows_backward, range(4))]

    # forward
    flows_forward = flow_net(x_2, x_1)
    flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                     zip(flows_forward, range(4))]

    return flows_backward, flows_forward

def get_flow_4frames(flows_forward, flows_backward):
    '''Get flow between t and t+2 from (t,t+1) and (t+1,t+2).'''

    # backward
    d = flows_forward[0].shape[1]
    flows_backward2 = []
    for flows in flows_backward:
        flow_list = []
        for i in range(d - 1, 0, -1):
            flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
            flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
            flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+2 to i
        flows_backward2.append(torch.stack(flow_list, 1))

    # forward
    flows_forward2 = []
    for flows in flows_forward:
        flow_list = []
        for i in range(1, d):
            flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
            flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
            flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-2 to i
        flows_forward2.append(torch.stack(flow_list, 1))

    return flows_backward2, flows_forward2

def get_flow_6frames(flows_forward, flows_backward, flows_forward2, flows_backward2):
    '''Get flow between t and t+3 from (t,t+2) and (t+2,t+3).'''

    # backward
    d = flows_forward2[0].shape[1]
    flows_backward3 = []
    for flows, flows2 in zip(flows_backward, flows_backward2):
        flow_list = []
        for i in range(d - 1, 0, -1):
            flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
            flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
            flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+3 to i
        flows_backward3.append(torch.stack(flow_list, 1))

    # forward
    flows_forward3 = []
    for flows, flows2 in zip(flows_forward, flows_forward2):
        flow_list = []
        for i in range(2, d + 1):
            flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
            flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
            flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-3 to i
        flows_forward3.append(torch.stack(flow_list, 1))

    return flows_backward3, flows_forward3


# -- network api --
def get_warped_frames(video,bflow,fflow,mode):
    if mode == "dnls":
        bwd, fwd = get_dnls_aligned(video,bflow,fflow)
    else:
        bwd, fwd = get_aligned_image_2frames(video,bflow,fflow)
    print("video.shape, bwd.shape,fwd.shape: ",video.shape,bwd.shape,fwd.shape)
    video = torch.cat([video, bwd, fwd], 2)
    return video

def get_dnls_aligned(video,bflow,fflow):
    fwd,bwd = dnls.nn.flow_patches_f.get_warp_2f(video,bflow,fflow)
    return bwd,fwd

def get_aligned_image_2frames(x, flows_backward, flows_forward):
    '''Parallel feature warping for 2 frames.'''

    # backward
    n = x.size(1)
    x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
    # print(len(x_backward[0].shape),len(flows_backward))
    # print(th.all(flows_backward[-1]==0))
    # print(th.all(flows_backward[0]==0))
    for i in range(n - 1, 0, -1):
        print(i,i-1,n)
        x_i = x[:, i, ...]
        flow = flows_backward[:, i - 1, ...]
        print(i,x_i.shape,flow.shape)
        x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i+1 aligned towards i
    # print(th.all(x_backward[-1]==0))
    print("len(x_backward): ",len(x_backward))
    print("len(x_backward): ",x_backward[0].shape)


    # forward
    x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
    for i in range(0, n - 1):
        x_i = x[:, i, ...]
        flow = flows_forward[:, i, ...]
        x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i-1 aligned towards i

    return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4': # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)

        out = torch.cat([output00, output01, output10, output11], 1)
        print("out.shape: ",out.shape)
        return out

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output
