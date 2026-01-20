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
void cuda_roc(const float* X, const int* all_kernel_data, float* y_ppv, float* y_max) {
    extern __shared__ int kernel_values[N_DATA_PER_KERNEL];

    const int id = blockIdx.y * KERNEL_AMOUNT + blockIdx.x;

    if (threadIdx.x < N_DATA_PER_KERNEL){
        kernel_values[threadIdx.x] = all_kernel_data[N_DATA_PER_KERNEL * blockIdx.x + threadIdx.x];
    }
    __syncthreads();

    int new_thread_idx = threadIdx.x;
    while (new_thread_idx < CALCS_PER_KERNEL) {
        float sum = *((float*)&BIAS);
        for (int j = 0; j < NUM_CHANNEL_INDICES; j++) {
            const int dim_idx = kernel_values[N_WEIGHTS_PER_KERNEL+j];
            for (int i = 0; i < KERNEL_LENGTH; i++) {
                const int series_value_idx = new_thread_idx - PADDING + DILATION * i;
                if (series_value_idx >= 0 && series_value_idx < SERIES_LENGTH) {
                    const unsigned int series_idx = blockIdx.y * (N_DIMENSIONS*SERIES_LENGTH) + dim_idx * SERIES_LENGTH + series_value_idx;
                    sum += *((float*)&kernel_values[j * KERNEL_LENGTH + i]) * X[series_idx];
                }
            }
        }
        if (sum > 0) atomicAdd_block(&PPV, 1);
        atomicMaxFloat_block(((float*)&MAX), sum);
        new_thread_idx += MAX_BLOCK_DIM_X;
    }
    __syncthreads();
    if (threadIdx.x == 0) y_ppv[id] = PPV;
    else if (threadIdx.x == 1) y_max[id] = *((float*)&MAX);
}

