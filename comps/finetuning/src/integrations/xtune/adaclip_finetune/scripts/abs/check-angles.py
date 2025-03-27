# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re

import torch
import torch.nn as nn

root = "abs-study/didemo-abs-16/2024-11-21_15-30-03"  # sys.argv[1]
p = re.compile(r"^clip-state-dict-\d{4}\.pth$")
pths = sorted([_ for _ in os.listdir(root) if p.match(_)])
cosine = nn.CosineSimilarity(dim=0)
sd = []
start_index = 3
pths = pths[start_index:]

for i, pth in enumerate(pths):
    loaded = torch.load(os.path.join(root, pth), map_location="cpu", weights_only=False)
    for k, v in loaded.items():
        loaded[k] = v.flatten()
    sd.append(loaded)


angles = {}
for i in range(1, len(pths)):
    angles[i] = (
        {k: torch.acos(cosine(sd[0][k], sd[i][k]).clamp_(-1, 1)) * 180 / torch.pi for k in sd[0]}
        if i > start_index
        else None
    )

last_index = len(pths) - 1
for k, v in angles[last_index].items():
    dim = sd[last_index][k].size()[0]
    print(f"{k: <48s}\t{dim:0>8d}\t{v: >6.2f}")

from matplotlib import pyplot

for i in range(start_index + 1, len(pths)):
    pyplot.figure(f"hist-{i}")
    pyplot.hist(list(angles[i].values()), 100)
    pyplot.savefig(os.path.join(root, pths[i] + ".hist.png"))
    pyplot.close(f"hist-{i}")

    dim_angle_pairs = [(sd[i][k].size()[0], angles[i][k]) for k in sd[i]]
    pyplot.figure(f"dim-corr-{i}")
    pyplot.semilogx([_[0] for _ in dim_angle_pairs], [_[1] for _ in dim_angle_pairs], "k.")
    pyplot.savefig(os.path.join(root, pths[i] + ".dim-corr.png"))
    pyplot.close(f"dim-corr-{i}")
