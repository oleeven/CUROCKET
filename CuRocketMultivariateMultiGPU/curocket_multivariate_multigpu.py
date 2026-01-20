import cupy as cp
import numpy as np
from time import time
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