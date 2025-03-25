# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from torch import nn

from llmfoundry.layers_registry import fcs
from llmfoundry.custom_layers import BandMatrix

fcs.register('torch', func=nn.Linear)

try:
    import transformer_engine.pytorch as te
    fcs.register('te', func=te.Linear)
except:
    pass

fcs.register('band', func=BandMatrix)
