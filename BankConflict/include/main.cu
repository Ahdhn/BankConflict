// credits
// https://github.com/Kobzol/hardware-effects-gpu/tree/master/bank-conflicts
#include <assert.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "helper.h"

constexpr uint32_t max_offset = 32;
constexpr uint32_t num_run    = 100;

template <typename T>
__global__ void kernel_1d(size_t shmem_size,
                          int    num_repeat,
                          T*     d_output,
                          int    offset)
{
    assert(shmem_size == offset * blockDim.x);

    extern __shared__ T s_mem[];

    for (int i = threadIdx.x; i < shmem_size; i++) {
        s_mem[i] = 0;
    }
    __syncthreads();


    int id = (threadIdx.x * offset) % shmem_size;
    for (int i = 0; i < num_repeat; i++) {
        s_mem[id] += id * i;
        id += 32;
        id %= shmem_size;
    }
    __syncthreads();

    d_output[blockIdx.x * blockDim.x + threadIdx.x] = s_mem[threadIdx.x];
}


void run_test_1d(size_t size, int offset, int num_repeat)
{
    using T = uint32_t;

    thrust::device_vector<T> d_input(size);

    const int block_size = 256;
    const int num_blocks = DIVIDE_UP(size, block_size);

    size_t shmem_size = block_size * max_offset;

    CUDA_ERROR(cudaProfilerStart());
    float sum_time = 0;
    for (int d = 0; d < num_run; ++d) {
        CUDATimer timer;
        timer.start();
        kernel_1d<<<num_blocks, block_size, shmem_size * sizeof(T)>>>(
            shmem_size, num_repeat, d_input.data().get(), offset);
        timer.stop();
        auto err = cudaDeviceSynchronize();
        EXPECT_EQ(err, cudaSuccess);
        sum_time += timer.elapsed_millis();
    }
    CUDA_ERROR(cudaProfilerStop());
    std::cout << "\n offset= " << offset
              << ", time(ms)= " << sum_time / float(num_run);
}

TEST(Test, run_1d)
{
    size_t size = 1024 * 1024;

    int num_repeat = 1;
    std::cout << "\n**** size= " << size << ", num_run = " << num_run;

    for (int offset = 1; offset <= max_offset; ++offset) {
        run_test(size, offset, num_repeat);
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
