# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch


"""
modified from UNITER codebase

A meta data loader for sampling from different datasets / training tasks
A prefetch loader to speedup data loading
"""


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_device(t, device) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_device(t, device) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_device(t, device) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_device_stream(batch, device):
    if isinstance(batch, torch.Tensor):
        if str(device) == "xpu":
            batch.record_stream(torch.xpu.current_stream(device=device))
        else:
            batch.record_stream(torch.cuda.current_stream(device=device))

    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_device_stream(t, device)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_device_stream(t, device)
    else:
        pass


class PrefetchLoader:
    """Overlap compute and device data transfer."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        if str(device) == "xpu":
            self.stream = torch.xpu.Stream(device=device)
        elif str(device) == "cuda":
            self.stream = torch.cuda.Stream()
        else:
            raise ValueError(f"Unsupported device type: {device}")

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            batch["clip_inputs"] = batch["clip_inputs"].float()
            batch["policy_inputs"] = batch["policy_inputs"].float() if batch["policy_inputs"] is not None else None
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        if str(self.device) == "xpu":
            with torch.xpu.stream(self.stream):
                self.batch = move_to_device(self.batch, self.device)
        else:
            with torch.cuda.stream(self.stream):
                self.batch = move_to_device(self.batch, self.device)
                # more code for the alternative if record_stream() doesn't work:
                # copy_ will record the use of the pinned source tensor in this
                # side stream.
                # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
                # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
                # self.next_input = self.next_input_gpu
                # self.next_target = self.next_target_gpu

    def next(self, it):
        if str(self.device) == "xpu":
            torch.xpu.current_stream(device=self.device).wait_stream(self.stream)
        else:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_device_stream(batch, self.device)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method
