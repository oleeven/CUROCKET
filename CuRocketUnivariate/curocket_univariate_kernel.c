struct blockDim{int x;};
struct blockIdx{int x;};
struct threadIdx{int x;};
blockDim;
blockIdx;
threadIdx;

//start
__device__ __forceinline__ float atomicMaxFloat_block(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax_block((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin_block((unsigned int*)addr, __float_as_uint(value)));

    return old;
}
extern "C" __global__
void cuda_rocket(const float* X, const int* all_kernel_data, float* y_ppv, float* y_max) {
    extern __shared__ int kernel_values[N_DATA_PER_KERNEL]; // shared kernel data within block

    const int kernel_idx = blockIdx.x; // index of kernel out of all kernels
    const int instance_idx = blockIdx.y; // index of instance within batch
    const int feature_idx = instance_idx * KERNEL_AMOUNT + kernel_idx; // index of feature

    // the first N_DATA_PER_KERNEL threads of the block read in one value of the kernel data each
    if (threadIdx.x < N_DATA_PER_KERNEL) kernel_values[threadIdx.x] = all_kernel_data[N_DATA_PER_KERNEL * kernel_idx + threadIdx.x];
    __syncthreads(); // all threads of the block wait for the data to be fully read in

    int dot_product_idx = threadIdx.x; // index of the current dot product
    while (dot_product_idx < CALCS_PER_KERNEL) { // loop while the current calculation is valid
        float sum = *((float*)&BIAS); // initiate sum with bias
        for (size_t i = 0; i < KERNEL_LENGTH; i++) { // loop through each calculation in the dot product
            const int series_value_idx = dot_product_idx - PADDING + DILATION * i; // calculate the index of the timepoint in the series
            if (series_value_idx >= 0 && series_value_idx < SERIES_LENGTH) { // check if index is within the series 
                float result = *((float*)&kernel_values[i]) * X[instance_idx * SERIES_LENGTH + series_value_idx]; // perform multiplication of kernel weight and value of timepoint
                sum += result; // add value to sum
            }
        }
        if (sum > 0) atomicAdd_block(&PPV, 1); // add 1 too ppv counter if sum is greater than 0
        atomicMaxFloat_block(((float*)&MAX), sum); // set new max with sum if sum if greater than current max
        dot_product_idx += MAX_BLOCK_DIM_X; // increase dot product index
    }
    __syncthreads(); // wait for all calulations in the block to be done
    if (threadIdx.x == 0) y_ppv[feature_idx] = PPV; // the first thread writes the ppv value from block cache to GPU RAM
    else if (threadIdx.x == 1) y_max[feature_idx] = *((float*)&MAX); // the second thread writes the max value from block cache to GPU RAM
}

