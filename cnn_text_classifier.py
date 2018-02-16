# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/switch_attractor_esn
# Description : NARMA 30 prediction with ESN.
# Date : 26th of January, 2018
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>


# Imports
import torch.utils.data
import dataset
from echotorch.transforms import text


# Author identification dataset
pan18loader = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=text.GloveVector(), problem=1),
    batch_size=1, shuffle=True)

# Get training data for this fold
for i, data in enumerate(pan18loader):
    # Inputs and labels
    inputs, labels = data
# end for
