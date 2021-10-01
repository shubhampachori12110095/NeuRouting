# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

from torch.autograd import Variable

from neurewriter.parser import parse_solution

_PAD = b"_PAD"

PAD_ID = 0
START_VOCAB_SIZE = 1
max_token_len = 5


def np_to_tensor(inp, output_type, cuda_flag, volatile_flag=False):
    if output_type == 'float':
        inp_tensor = Variable(torch.FloatTensor(inp), requires_grad=volatile_flag)
    elif output_type == 'int':
        inp_tensor = Variable(torch.LongTensor(inp), requires_grad=volatile_flag)
    else:
        print('undefined tensor type')
    if cuda_flag:
        inp_tensor = inp_tensor.cuda()
    return inp_tensor


def process_batch(data, batch_size, start_idx=None):
    data_size = len(data)
    if start_idx is not None:
        batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + batch_size))]
    else:
        batch_idxes = np.random.choice(len(data), batch_size)
    batch_data = []
    for idx in batch_idxes:
        problem = data[idx]
        dm = parse_solution(problem)
        batch_data.append(dm)
    return batch_data
