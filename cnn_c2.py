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
import random

# Experience parameter
window_size = 500
batch_size = 64
sample_batch = 4
epoch_batches = 10000
max_epoch = 300

# Author identification dataset
pan18loader = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=text.Character2Gram(), problem=1),
    batch_size=1, shuffle=True)

# Total training data
training_data = list()
training_labels = list()

# Get training data
for i, data in enumerate(pan18loader):
    # Inputs and labels
    inputs, labels = data

    # Add
    training_data.append(inputs)
    training_labels.append(labels)
# end for

# Number of samples
n_samples = len(training_labels)

# For each iteration
for epoch in range(max_epoch):
    # For each batch
    for b in range(epoch_batches):
        # Get samples for the batch
        for i in range(batch_size):
            # Random sample and position
            random_sample = random.randint(0, n_samples-1)
            random_sample_size = training_data[random_sample].size(1)
            random_position = random.randint(0, random_sample_size-window_size-1)
            sample = training_data[random_sample]

            # Get sequence
            random_sequence = sample[:, random_position:random_position+window_size]

            # Append
            if i == 0:
                batch = random_sequence
            else:
                batch = torch.cat((batch, random_sequence), dim=0)
            # end if
        # end for

        # Print
        print(batch.size())
    # end for
# end for
