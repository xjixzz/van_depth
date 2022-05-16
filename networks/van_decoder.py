# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class VANDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=1, use_skips=True):
        super(VANDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 128, 320])

        # decoder
        self.convs = OrderedDict()
        for i in range(3, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 3 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in range(3, -1, -1):
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(3, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            
            if i==0:
                x = upsample(x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i==0:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                #print(self.outputs[("disp", i)].shape)
            else:
                self.outputs[("disp", i+1)] = self.sigmoid(self.convs[("dispconv", i)](x))
                #print(self.outputs[("disp", i+1)].shape)

        return self.outputs
