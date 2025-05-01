from itertools import repeat
import collections.abc
from typing import Callable
import jittor.nn as nn

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

def apply_recursive(fn: Callable, module: nn.Module, depth_first: bool = True, include_root: bool = False):
    if not depth_first and include_root:
        fn(module)

    child_iterator = module.children()

    for child_module in child_iterator:
        apply_recursive(fn=fn, module=child_module, depth_first=depth_first, include_root=True)

    if depth_first and include_root:
        fn(module)

    return module


def accuracy(output, target, topk = (1, )):

    maxk = min(output.shape[1], max(topk))
    batch_size = target.shape[0]

    _, prediction = output.topk(maxk, dim = 1)
    target = target.unsqueeze(0).permute(0, 1)
    correct = prediction.equal(target.expand_as(prediction))

    return [correct[:,:min(k, maxk)].any(dim = 1).sum() * 100. / batch_size for k in topk]

def index_fill_inplace(x, dim, index, value):
    indices = [slice(None)] * x.ndim
    indices[dim] = index
    index_tuple = tuple(indices)
    x[index_tuple] = value


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple