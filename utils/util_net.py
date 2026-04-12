#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 20:29:36

import math
import torch
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import torch.nn.functional as F

def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = F.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad

def forward_chop(net, x, net_kwargs=None, scale=1, shave=10, min_size=160000):
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if net_kwargs is None:
                sr_batch = net(lr_batch)
            else:
                sr_batch = net(lr_batch, **net_kwargs)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def measure_time(net, inputs, num_forward=100):
    '''
    Measuring the average runing time (seconds) for pytorch.
    out = net(*inputs)
    '''
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.set_grad_enabled(False):
        for _ in range(num_forward):
            out = net(*inputs)
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / 1000

def reload_model(model, ckpt, strict=True):
    """
    Reload model weights from checkpoint dict.

    strict=True: require every model key to exist in ckpt (legacy behavior).
    strict=False: load matched keys only, skip missing/shape-mismatch keys.
    """
    if not isinstance(ckpt, dict) or len(ckpt) == 0:
        raise AssertionError('Invalid checkpoint state_dict')

    first_key = list(ckpt.keys())[0]
    module_flag = first_key.startswith('module.')
    compile_flag = '_orig_mod' in first_key

    missing_keys = []
    shape_mismatch = []
    loaded = 0

    for source_key, source_value in model.state_dict().items():
        target_key = source_key
        if compile_flag and (not '_orig_mod.' in source_key):
            target_key = '_orig_mod.' + target_key
        if module_flag and (not source_key.startswith('module')):
            target_key = 'module.' + target_key

        if target_key not in ckpt:
            missing_keys.append(source_key)
            continue

        target_value = ckpt[target_key]
        if source_value.shape != target_value.shape:
            shape_mismatch.append((source_key, tuple(source_value.shape), tuple(target_value.shape)))
            continue

        source_value.copy_(target_value)
        loaded += 1

    if strict and (missing_keys or shape_mismatch):
        parts = []
        if missing_keys:
            parts.append(f"missing={len(missing_keys)} first={missing_keys[0]}")
        if shape_mismatch:
            k, s1, s2 = shape_mismatch[0]
            parts.append(f"shape_mismatch={len(shape_mismatch)} first={k} model={s1} ckpt={s2}")
        raise AssertionError('reload_model strict load failed: ' + '; '.join(parts))

    return {
        'loaded': loaded,
        'missing': missing_keys,
        'shape_mismatch': shape_mismatch,
    }
