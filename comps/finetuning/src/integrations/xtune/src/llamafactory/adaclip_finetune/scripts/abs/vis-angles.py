# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch


def visualize(n, dim, d):
    rw = torch.randn(1, dim)
    dw = d * torch.randn(n, dim)

    cosine = torch.nn.CosineSimilarity(dim=1)

    angles = torch.acos(cosine(rw, rw + dw).clamp_(-1, 1)) * 180 / torch.pi
    print("{: > 4d}\t{:.4f}\t{:.4g}\t{:.4g}".format(dim, d, angles.mean().item(), angles.std().item()))

    from matplotlib import pyplot

    output_name = f"angle-hist-N{n}-D{dim}-d{d}"
    pyplot.figure(output_name)
    pyplot.title(output_name)
    pyplot.hist(angles, 100)
    pyplot.savefig(output_name + ".png")
    pyplot.close(output_name)


if __name__ == "__main__":

    n_list = [
        10000,
    ]
    dim_list = [1024, 2048, 4096, 8192]
    d_list = [
        0.05,
    ]  # 0.1, 0.33]
    for n in n_list:
        for dim in dim_list:
            # print(scipy.stats.chi.mean(dim), scipy.stats.chi2.mean(dim), torch.randn(n, dim).norm(dim=1).mean())
            for d in d_list:
                visualize(n, dim, d)
