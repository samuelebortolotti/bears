from functools import reduce
from itertools import chain

import numpy as np
import torch


def squeeze_last_dimensions(arr, numpy=True):
    if not numpy:
        while torch.any(torch.tensor(arr.shape[-1:]) == 1):
            arr = torch.squeeze(arr, dim=-1)
        return arr
    else:
        while np.any(np.array(arr.shape[-1:]) == 1):
            arr = np.squeeze(arr, axis=-1)
        return arr


def convert_np_array_to_binary(array):
    return np.apply_along_axis(
        lambda row: int("".join(map(str, row[::-1])), 2),
        axis=1,
        arr=array.astype(int),
    )


def compute_world_probability(
    concepts_list, numpy=True, clip_value=1e-5
):
    if not numpy:
        world_matrix = squeeze_last_dimensions(
            concepts_list[0], numpy
        )
        for tensor in concepts_list[1:]:
            world_matrix = world_matrix.unsqueeze(-1)
            world_matrix = torch.matmul(
                world_matrix, squeeze_last_dimensions(tensor, numpy)
            )

        collapsed_dim = np.prod(world_matrix.shape[2:])
        world_matrix = world_matrix.view(
            world_matrix.shape[0],
            world_matrix.shape[1],
            collapsed_dim,
        )

        world_matrix = torch.mean(world_matrix, dim=0)

        return world_matrix

    else:

        # Initialize the result with the first tensor
        world_matrix = squeeze_last_dimensions(concepts_list[0])

        # Multiply each tensor with the previous result -> final shape [30, 512, 2, 2, 2..., 2]
        for tensor in concepts_list[1:]:
            world_matrix = np.expand_dims(world_matrix, axis=-1)
            world_matrix = np.matmul(
                world_matrix,
                squeeze_last_dimensions(tensor),
            )

        collapsed_dim = np.prod(world_matrix.shape[2:])
        world_matrix = np.reshape(
            world_matrix,
            (
                world_matrix.shape[0],
                world_matrix.shape[1],
                collapsed_dim,
            ),
        )

        # mean across the num_models
        world_matrix = np.mean(world_matrix, axis=0)

    return world_matrix


def create_concepts_array(concepts_list, num_ones, i=2, numpy=True):
    n_models = concepts_list.shape[0]
    batch_size = concepts_list.shape[1]

    shape = (
        (n_models, batch_size)
        + ((1,) * i)
        + (2,)
        + ((1,) * (num_ones - i))
    )

    slice_1 = tuple(
        [slice(None), slice(None)]
        + [0] * i
        + [0]
        + [0] * (num_ones - i)
    )
    slice_2 = tuple(
        [slice(None), slice(None)]
        + [0] * i
        + [1]
        + [0] * (num_ones - i)
    )

    c_array = np.zeros(shape)

    if not numpy:
        c_array = torch.zeros(shape)

    c_array[slice_1] = concepts_list[:, :, i]
    c_array[slice_2] = 1 - concepts_list[:, :, i]

    return c_array


def compute_forward_stop_prob(concepts, numpy=True):
    # concepts: models, batch, 1
    concepts_list = []
    for i in range(9):
        c_array = create_concepts_array(concepts, 8, i, numpy)
        concepts_list.append(c_array)
    return compute_world_probability(concepts_list, numpy)


def compute_forward_prob(concepts, numpy=True):
    # concepts: models, batch, 1
    concepts_list = []
    for i in range(3):
        c_array = create_concepts_array(concepts, 2, i, numpy)
        concepts_list.append(c_array)
    return compute_world_probability(concepts_list, numpy)


def compute_stop_prob(concepts, numpy=True):
    # concepts: models, batch, 1
    concepts_list = []
    for i in range(6):
        c_array = create_concepts_array(concepts, 5, i, numpy)
        concepts_list.append(c_array)
    return compute_world_probability(concepts_list, numpy)


def compute_forward_groundtruth(groundtruth):
    return convert_np_array_to_binary(groundtruth[:, :3])


def compute_stop_groundtruth(groundtruth):
    return convert_np_array_to_binary(groundtruth[:, 3:9])


def compute_forward_stop_groundtruth(groundtruth):
    return convert_np_array_to_binary(groundtruth[:, :9])


def compute_left(concepts, numpy=True):
    # concepts: models, batch, 1
    concepts_list = []
    for i in range(9, 15):
        c_array = create_concepts_array(concepts, 5, i, numpy)
        concepts_list.append(c_array)
    return compute_world_probability(concepts_list, numpy)


def compute_left_groundtruth(groundtruth):
    return convert_np_array_to_binary(groundtruth[:, 9:15])


def compute_right(concepts, numpy=True):
    # concepts: models, batch, 1
    concepts_list = []
    for i in range(15, 21):
        c_array = create_concepts_array(concepts, 5, i, numpy)
        concepts_list.append(c_array)
    return compute_world_probability(concepts_list, numpy)


def compute_right_groundtruth(groundtruth):
    return convert_np_array_to_binary(groundtruth[:, 15:21])


def compute_output_probability(outs):
    return np.mean(outs, axis=0)
