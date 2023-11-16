# %%
import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part2_cnns', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")
arr.shape
# %%
display_array_as_img(arr[0])
# %%
display_soln_array_as_img(1)
# %%
arr1 = einops.rearrange(arr, "b c h w -> c h (b w)")
display_array_as_img(arr1)
# %%
display_soln_array_as_img(2)
# %%
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
display_array_as_img(arr2)
# %%
display_soln_array_as_img(3)
# %%
arr3 = einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)
# %%
display_soln_array_as_img(9)
arr9 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr9)
# %%
display_soln_array_as_img(10)
arr10 = einops.reduce(arr, "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", 'max', b1=2)
display_array_as_img(arr10)
# %%
def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, "i i->")

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, "i j,j->i")

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, "i j, j k -> i k")

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, "i,i->")

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, "i,j->i j")

tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)

# %%
test_input = t.tensor(
    [[0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19]], dtype=t.float
)
# %%
test_input.stride()
# %%
import torch as t
from collections import namedtuple

TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(
        output=t.tensor([0, 1, 2, 3]),
        size=(4,),
        stride=(1,),
    ),
    TestCase(
        output=t.tensor([[0, 2], [5, 7]]),
        size=(2, 2),
        stride=(5, 2),
    ),

    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]),
        size=(5,),
        stride=(1,),
    ),

    TestCase(
        output=t.tensor([0, 5, 10, 15]),
        size=(5,),
        stride=(5,),
    ),

    TestCase(
        output=t.tensor([
            [0, 1, 2],
            [5, 6, 7]
        ]),
        size=(2,3,),
        stride=(5,1,),
    ),

    TestCase(
        output=t.tensor([
            [0, 1, 2],
            [10, 11, 12]
        ]),
        size=(2,3,),
        stride=(10,1),
    ),

    TestCase(
        output=t.tensor([
            [0, 0, 0],
            [11, 11, 11]
        ]),
        size=None,
        stride=None,
    ),

    TestCase(
        output=t.tensor([0, 6, 12, 18]),
        size=None,
        stride=None,
    ),
]

for (i, test_case) in enumerate(test_cases):
    if (test_case.size is None) or (test_case.stride is None):
        print(f"Test {i} failed: attempt missing.")
    else:
        actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
        if (test_case.output != actual).any():
            print(f"Test {i} failed:")
            print(f"Expected: {test_case.output}")
            print(f"Actual: {actual}\n")
        else:
            print(f"Test {i} passed!\n")
# %%
test_input.as_strided(size=(2,3,), stride=(5,1))
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
