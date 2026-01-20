import cupy as cp
import numpy as np
from time import time

def _apply_kernels_univariate(X, kernels):
    (
    kernel_weights,
    kernel_lengths,
    biases,
    dilations,
    paddings,
    num_channel_indices,
    channel_indices,
    ) = kernels
        
    n_instances, n_dimensions, n_timepoints = X.shape
        
    n_kernels = len(kernel_lengths)

    cuda_device = cp.cuda.Device()
    max_grid_dim_x = cuda_device.attributes["MaxGridDimX"]
    assert n_kernels < max_grid_dim_x, f"Too many kernels. Max amount of kernels is {max_grid_dim_x}"
    max_grid_dim_y = cuda_device.attributes["MaxGridDimY"]
    max_block_dim_x = cuda_device.attributes["MaxBlockDimX"]
    mem_info = cuda_device.mem_info
    available_size = mem_info[0] / 1e9
    total_size = mem_info[1] / 1e9
    # print(f"Note: {(available_size/total_size*100):.1f}% of graphics memory are currently free. Free as much memory as possible to make the algorithm work most efficiently.")
    input_size = 4 * (n_timepoints*n_instances) / 1e9
    kernel_size = 4 * (18*n_kernels) / 1e9
    output_size = 4 * (2 * n_kernels * n_instances) / 1e9 
    needed_size = input_size + kernel_size + output_size
    free_size = available_size - kernel_size
    n_instances_optimal = int(free_size / (4 * (n_timepoints + 2 * n_kernels) / 1e9))
    if n_instances_optimal > n_instances: n_instances_optimal = n_instances
    if n_instances_optimal > max_grid_dim_y: n_instances_optimal = max_grid_dim_y

    kernel_starts = np.concatenate(([0], np.cumsum(kernel_lengths)[:-1])) 
    calcs_per_kernel = n_timepoints + 2 * paddings - dilations * (kernel_lengths-1)
    y_ppv = np.empty((n_instances, n_kernels), np.float32)
    y_max = np.empty((n_instances, n_kernels), np.float32)
    y_ppv_ = cp.empty((n_instances_optimal, n_kernels), cp.float32)
    y_max_ = cp.empty((n_instances_optimal, n_kernels), cp.float32)

    def float_arr_2_int(arr): return np.frombuffer(arr.tobytes(), np.int32)

    n_data_per_kernel = 18
    all_kernel_data = np.empty((n_kernels, n_data_per_kernel), np.int32)
    for i in range(n_kernels):
        start = kernel_starts[i] 
        length = kernel_lengths[i]
        all_kernel_data[i, 0:length] = float_arr_2_int(kernel_weights[start:start+length]) # 0:11 reserved for kernel weights
        all_kernel_data[i, 11] = length # 11 kernel length
        all_kernel_data[i, 12:13] = float_arr_2_int(biases[i]) # 12 bias
        all_kernel_data[i, 13] = dilations[i] # 13 dilation
        all_kernel_data[i, 14] = paddings[i] # 14 padding
        all_kernel_data[i, 15] = calcs_per_kernel[i] # 15 calculations per kernel
        all_kernel_data[i, 16] = 0 # 16 ppv
        all_kernel_data[i, 17:18] = float_arr_2_int(np.float32(-np.inf)) # 17 max
    all_kernel_data_ = cp.array(all_kernel_data, dtype=cp.int32)


    with open("CuRocketUnivariate/curocket_univariate_kernel.c", "r") as f:
        startword = "//start"
        kernel_string = f.read()
        start_idx = kernel_string.find(startword) + len(startword)
        kernel_string = kernel_string[start_idx:]

        to_replace = (("MAX_BLOCK_DIM_X", max_block_dim_x),
                      ("SERIES_LENGTH", n_timepoints),
                      ("N_DIMENSIONS", n_dimensions),
                      ("KERNEL_AMOUNT", n_kernels),
                      ("N_DATA_PER_KERNEL", n_data_per_kernel),
        )

        for name, value in to_replace:
            kernel_string = kernel_string.replace(name, str(value))

        kernel_attributes = ("KERNEL_LENGTH",
                             "BIAS",
                             "DILATION",
                             "PADDING",
                             "CALCS_PER_KERNEL",
                             "PPV",
                             "MAX")
        for i, k in enumerate(kernel_attributes):
            kernel_string = kernel_string.replace(k, f"kernel_values[{11+i}]")


    cuda_roc = cp.RawKernel(kernel_string, "cuda_rocket")
    cuda_roc.compile()

    series_iterations = int((n_instances-1) / n_instances_optimal)+1
    series_start_idx = 0
    series_end_idx = series_start_idx + n_instances_optimal
    series_amount_iteration = n_instances_optimal

    for series_iteration in range(series_iterations):
        X_ = cp.array(X[series_start_idx:series_end_idx], dtype=cp.float32)

        cuda_roc((n_kernels, series_amount_iteration), (max_block_dim_x,), (X_, all_kernel_data_, y_ppv_, y_max_))
        cp.cuda.get_current_stream().synchronize()

        y_ppv[series_start_idx:series_end_idx, :] = y_ppv_.get()[:series_amount_iteration, :] / calcs_per_kernel
        y_max[series_start_idx:series_end_idx, :] = y_max_.get()[:series_amount_iteration, :]
        del X_
        cp.get_default_memory_pool().free_all_blocks()

        series_start_idx += n_instances_optimal
        series_end_idx += n_instances_optimal
        if series_end_idx > n_instances: series_end_idx = n_instances
        series_amount_iteration = series_end_idx - series_start_idx

    del all_kernel_data_, y_ppv_, y_max_, cuda_roc
    cp.get_default_memory_pool().free_all_blocks()


    merged_array = np.empty((n_instances, n_kernels*2), np.float32)
    merged_array[:, 0::2] = y_ppv  # Place y_ppv into even columns
    merged_array[:, 1::2] = y_max  # Place y_max into odd columns
    return merged_array

