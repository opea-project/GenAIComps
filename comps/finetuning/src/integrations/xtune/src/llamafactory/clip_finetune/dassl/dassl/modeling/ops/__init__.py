# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .attention import *
from .conv import *
from .cross_entropy import cross_entropy
from .dsbn import DSBN1d, DSBN2d
from .efdmix import (
    EFDMix,
    activate_efdmix,
    crossdomain_efdmix,
    deactivate_efdmix,
    random_efdmix,
    run_with_efdmix,
    run_without_efdmix,
)
from .mixstyle import (
    MixStyle,
    activate_mixstyle,
    crossdomain_mixstyle,
    deactivate_mixstyle,
    random_mixstyle,
    run_with_mixstyle,
    run_without_mixstyle,
)
from .mixup import mixup
from .mmd import MaximumMeanDiscrepancy
from .optimal_transport import MinibatchEnergyDistance, SinkhornDivergence
from .reverse_grad import ReverseGrad
from .sequential2 import Sequential2
from .transnorm import TransNorm1d, TransNorm2d
