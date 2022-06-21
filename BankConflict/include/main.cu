//credits https://github.com/Kobzol/hardware-effects-gpu/tree/master/bank-conflicts
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "helper.h"

template <typename T>
__global__ void exec_kernel(size_t shmem_size,
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

 
    int id = threadIdx.x * offset;
    for (int i = 0; i < num_repeat; i++) {
        s_mem[id] += id * i;
        id += 32;
        id %= shmem_size;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < blockDim.x; i++) {
        d_output[blockIdx.x * blockDim.x + i] = s_mem[i];
    }
}


void run_test(size_t size, int offset, int num_repeat)
{
    using T = uint32_t;

    thrust::device_vector<T> d_input(size);

    const int block_size = 256;
    const int num_blocks = DIVIDE_UP(size, block_size);

    size_t shmem_size = block_size * offset;

    CUDATimer timer;
    timer.start();
    exec_kernel<<<num_blocks, block_size, shmem_size * sizeof(T)>>>(
        shmem_size, num_repeat, d_input.data().get(), offset);
    timer.stop();
    std::cout << "\n size= " << size << " , offset= " << offset
              << " , time(ms)= " << timer.elapsed_millis();
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

TEST(Test, exe)
{
    size_t size = 1024 * 1024;

    int num_repeat = 1;

    for (int offset = 1; offset <= 32; ++offset) {
        run_test(size, offset, num_repeat);
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
