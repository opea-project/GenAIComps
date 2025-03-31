# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from matplotlib import pyplot

loaded = torch.load("pre-trained/didemo-c-32-16.pth", map_location="cpu", weights_only=False)

stats = []
for n, p in loaded["state_dict"].items():
    if len(p.shape) == 2 and (n.startswith("clip.visual.transformer") or n.startswith("clip.transformer")):
        pyplot.figure(n)
        pyplot.title(n)
        pyplot.hist(p.flatten().numpy(), 100)
        pyplot.savefig(f"{n}_hist.png")
        pyplot.close()

        l = torch.sqrt(p.flatten().square().sum()).item()
        stats.append((p.numel(), l))

pyplot.figure("dim-dep")
pyplot.title("dim-dep: norm vs. sqrt(dim)")
pyplot.plot([math.sqrt(_[0]) for _ in stats], [_[1] for _ in stats], ".")
pyplot.savefig("dim-dep.png")
pyplot.close("dim-dep")
