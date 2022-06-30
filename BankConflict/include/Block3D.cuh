#pragma once
#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "helper.h"

template <int xSize, int ySize, int zSize, int radius, typename T>
__global__ void kernel_3d(T* d_output, int num_repeat)
{
    constexpr int shmem_size =
        (xSize + radius) * (ySize + radius) * (zSize + radius);

    // clang-format off
    const int read_id = threadIdx.x +
                        threadIdx.y * xSize +
                        threadIdx.z * xSize * ySize;
   
    const int block_id = blockIdx.x + 
                         blockIdx.y * blockDim.x +
                         blockIdx.z * blockDim.x * blockDim.y;

     int write_id = (threadIdx.x + radius) +
                    (threadIdx.y + radius) * xSize +
                    (threadIdx.z + radius) * xSize * ySize;
    // clang-format on

    __shared__ T s_mem[shmem_size];

    for (int i = read_id; i < shmem_size;
         i += blockDim.x * blockDim.y * blockDim.z) {
        s_mem[i] = 0;
    }
    __syncthreads();

    for (int i = 0; i < num_repeat; i++) {
        s_mem[write_id] += write_id + i;
        write_id += 32;
        write_id %= shmem_size;
    }

    __syncthreads();

    d_output[block_id + read_id] = s_mem[read_id];
}

template <int radius>
void run_test_3d(size_t size_1d, int num_repeat, int num_run)
{
    using T             = uint32_t;
    constexpr int xSize = 8;
    constexpr int ySize = 8;
    constexpr int zSize = 8;

    assert((size_1d % xSize) == 0 && (size_1d % ySize) == 0 &&
           (size_1d % zSize) == 0);

    thrust::device_vector<T> d_input(size_1d * size_1d * size_1d);

    CUDA_ERROR(cudaProfilerStart());
    float sum_time = 0;
    for (int d = 0; d < num_run; ++d) {
        CUDATimer timer;
        timer.start();
        kernel_3d<xSize, ySize, zSize, radius>
            <<<dim3(size_1d / xSize, size_1d / ySize, size_1d / zSize),
               dim3(xSize, ySize, zSize)>>>(d_input.data().get(), num_repeat);
        timer.stop();
        auto err = cudaDeviceSynchronize();
        EXPECT_EQ(err, cudaSuccess);
        sum_time += timer.elapsed_millis();
    }
    CUDA_ERROR(cudaProfilerStop());

    std::cout << "\n**** size= " << size_1d * size_1d * size_1d
              << ", num_run = " << num_run << ", radius = " << radius
              << ", time(ms)= " << sum_time / float(num_run);
}

TEST(Test, run_3d)
{
    const size_t size       = 1024;
    const int    num_run    = 100;
    const int    num_repeat = 1;
    run_test_3d<0>(size, num_repeat, num_run);
    //run_test_3d<1>(size, num_repeat, num_run);
    //run_test_3d<2>(size, num_repeat, num_run);
    printf("\n");
}