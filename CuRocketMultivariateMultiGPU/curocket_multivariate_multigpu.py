import cupy as cp
import numpy as np
from time import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from CuRocketMultivariate.curocket_multivariate import _apply_kernels_multivariate

def _apply_kernels(X, kernels):
    n_instances, n_dimensions, n_timepoints = X.shape

    device_names = cuda_init(0)
    assert len(set(device_names)) == 1, "in this implementation of cuda rocket all gpus must be of the same type"
    n_devices = len(device_names)

    n_instances_device_init = n_instances // n_devices
    rest = n_instances % n_devices
    n_instances_device = [n_instances_device_init] * n_devices
    for i in range(rest):
        n_instances_device[i] += 1
    n_instances_device_cumsum = np.concatenate([[0], np.cumsum(n_instances_device)])

    if n_instances <= 1:
        n_devices = 1

    iter_list = []
    for device_idx in range(n_devices):
        X_dev = X[n_instances_device_cumsum[device_idx]:n_instances_device_cumsum[device_idx + 1]]
        iter_list.append((X_dev, kernels, device_idx))

    results = []
    with ThreadPoolExecutor(max_workers=n_devices) as executor:
        futures = [executor.submit(_apply_kernels_multivariate, *args) for args in iter_list]
        for future in futures:
            results.append(future.result())

    merged_array = np.concatenate(results, 0)

    return merged_array

def cuda_init(_):
    device_count = cp.cuda.runtime.getDeviceCount()
    device_names = [cp.cuda.runtime.getDeviceProperties(device_id)["name"] for device_id in range(device_count)]
    return device_names

# currently unused, as the original _generate_kernels from sktime is used
def _generate_kernels(n_timepoints, num_kernels, n_columns, seed, kernel_lengths):
    if seed is not None:
        np.random.seed(seed)

    # candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    candidate_lengths = np.arange(*kernel_lengths, dtype=np.int32) #O
    print("cuda", candidate_lengths)
    lengths = np.random.choice(candidate_lengths, num_kernels).astype(np.int32)

    num_channel_indices = np.zeros(num_kernels, dtype=np.int32)
    for i in range(num_kernels):
        # limit = min(n_columns, lengths[i]) 
        limit = n_columns #O
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros(
        np.int32(
            np.dot(lengths.astype(np.float32), num_channel_indices.astype(np.float32))
        ),
        dtype=np.float32,
    )
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0  # for weights
    a2 = 0  # for channel_indices

    for i in range(num_kernels):
        _length = lengths[i]
        _num_channel_indices = num_channel_indices[i]

        _weights = np.random.normal(0, 1, _num_channel_indices * _length).astype(
            np.float32
        )

        b1 = a1 + (_num_channel_indices * _length)
        b2 = a2 + _num_channel_indices

        a3 = 0  # for weights (per channel)
        for _ in range(_num_channel_indices):
            b3 = a3 + _length
            _weights[a3:b3] = _weights[a3:b3] - _weights[a3:b3].mean()
            a3 = b3

        weights[a1:b1] = _weights

        channel_indices[a2:b2] = np.random.choice(
            np.arange(0, n_columns), _num_channel_indices, replace=False
        )

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(
            0, np.log2((n_timepoints - 1) / (_length - 1))
        )
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1
        a2 = b2

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )
