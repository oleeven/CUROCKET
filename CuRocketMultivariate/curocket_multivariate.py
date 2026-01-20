import cupy as cp
import numpy as np
from time import time

def _apply_kernels_multivariate(X, kernels, device_id = 0):
    (
    kernel_weights,
    kernel_lengths,
    bias,
    dilation,
    padding,
    num_channel_indices,
    channel_indices,
    ) = kernels
        
    n_instances, n_dimensions, n_timepoints = X.shape
        
    n_kernels = len(kernel_lengths)

    max_kernel_height = num_channel_indices.max()
    max_kernel_width = kernel_lengths.max()
    n_weights_per_kernel = max_kernel_height * max_kernel_width
    n_data_channels_per_kernel = max_kernel_height
    b = n_weights_per_kernel + n_data_channels_per_kernel
    n_data_per_kernel = b + 8
    all_kernel_data = np.empty((n_kernels, n_data_per_kernel), np.int32)

    # print(f"with device_id {device_id}")
    cuda_device = cp.cuda.Device(device_id)
    with cp.cuda.Device(device_id):

        max_grid_dim_x = cuda_device.attributes["MaxGridDimX"]
        assert n_kernels < max_grid_dim_x, f"Too manny kernels. Max amount of kernels is {max_grid_dim_x}"
        max_grid_dim_y = cuda_device.attributes["MaxGridDimY"]
        max_block_dim_x = cuda_device.attributes["MaxBlockDimX"]
        mem_info = cuda_device.mem_info
        available_size = mem_info[0] / 1e9
        total_size = mem_info[1] / 1e9
        # print(f"Note: {(available_size/total_size*100):.1f}% of graphics memory are currently free. Free as much memory as possible to make the algorithm work most efficiently.")
        bytes_per_number = 4
        kernel_size = bytes_per_number * (n_data_per_kernel*n_kernels) / 1e9
        free_size = available_size - kernel_size
        n_instances_optimal = int(free_size / (bytes_per_number * (n_timepoints*n_dimensions + 2 * n_kernels) / 1e9))
        if n_instances_optimal > n_instances: n_instances_optimal = n_instances
        if n_instances_optimal > max_grid_dim_y: n_instances_optimal = max_grid_dim_y

        def float_arr_2_int(arr): return np.frombuffer(arr.tobytes(), np.int32)

        whole_kernel_lengths = kernel_lengths * num_channel_indices
        kernel_starts = np.concatenate(([0], np.cumsum(whole_kernel_lengths)), dtype=np.int32)
        calcs_per_kernel = (n_timepoints + 2 * padding - dilation * (kernel_lengths-1))
        num_channel_indices_cumsum = np.concatenate(([0], np.cumsum(num_channel_indices)), dtype=np.int32)
        y_ppv = np.empty((n_instances, n_kernels), np.float32)
        y_max = np.empty((n_instances, n_kernels), np.float32)
        y_ppv_ = cp.empty((n_instances_optimal, n_kernels), cp.float32)
        y_max_ = cp.empty((n_instances_optimal, n_kernels), cp.float32)

        for i in range(n_kernels):
            start = kernel_starts[i] 
            whole_kernel_length = whole_kernel_lengths[i]
            all_kernel_data[i, 0:whole_kernel_length] = float_arr_2_int(kernel_weights[start:start+whole_kernel_length])
            all_kernel_data[i, n_weights_per_kernel:n_weights_per_kernel+num_channel_indices[i]] = channel_indices[num_channel_indices_cumsum[i]:num_channel_indices_cumsum[i+1]]
            all_kernel_data[i, b] = kernel_lengths[i]
            all_kernel_data[i, b+1] = num_channel_indices[i]
            all_kernel_data[i, b+2] = float_arr_2_int(bias[i])[0]
            all_kernel_data[i, b+3] = dilation[i]
            all_kernel_data[i, b+4] = padding[i]
            all_kernel_data[i, b+5] = calcs_per_kernel[i]
            all_kernel_data[i, b+6] = 0
            all_kernel_data[i, b+7] = float_arr_2_int(np.float32(-np.inf))[0]
        all_kernel_data_ = cp.array(all_kernel_data, dtype=cp.int32)

        with open("CuRocketMultivariate/curocket_multivariate_kernel.c", "r") as f:
            startword = "//start"
            kernel_string = f.read()
            start_idx = kernel_string.find(startword) + len(startword)
            kernel_string = kernel_string[start_idx:]

            to_replace = (("MAX_BLOCK_DIM_X", max_block_dim_x),
                        ("SERIES_LENGTH", n_timepoints),
                        ("N_DIMENSIONS", n_dimensions),
                        ("KERNEL_AMOUNT", n_kernels),
                        ("N_DATA_PER_KERNEL", n_data_per_kernel),
                        ("N_WEIGHTS_PER_KERNEL", n_weights_per_kernel),
                        ("MAX_KERNEL_WIDTH", max_kernel_width),
            )
            for name, value in to_replace:
                kernel_string = kernel_string.replace(name, str(value))

            kernel_attributes = ("KERNEL_LENGTH",
                                "NUM_CHANNEL_INDICES",
                                "BIAS",
                                "DILATION",
                                "PADDING",
                                "CALCS_PER_KERNEL",
                                "PPV",
                                "MAX")
            for i, k in enumerate(kernel_attributes):
                kernel_string = kernel_string.replace(k, f"kernel_values[{b+i}]")

        cuda_roc = cp.RawKernel(kernel_string, "cuda_roc")
        cuda_roc.compile()

        series_iterations = int((n_instances-1) / n_instances_optimal)+1
        series_start_idx = 0
        series_end_idx = series_start_idx + n_instances_optimal
        series_amount_iteration = n_instances_optimal


        for series_iteration in range(series_iterations):
            X_ = cp.array(X[series_start_idx:series_end_idx], dtype=cp.float32)
            cuda_roc((n_kernels, series_amount_iteration), (max_block_dim_x,),
                (X_, all_kernel_data_, y_ppv_, y_max_))
            cp.cuda.get_current_stream().synchronize()

            t = time()
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

