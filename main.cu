#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

template<const int TILE_SIZE>
__global__ void matrixMultiplyTiled(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {

    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        const int tileCol = tile * TILE_SIZE + threadIdx.x;
        const int tileRow = tile * TILE_SIZE + threadIdx.y;

        if (row < M && tileCol < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tileCol]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tileRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[tileRow * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

template<int BLOCK_SIZE, int THREAD_ITEMS_X=2, int THREAD_ITEMS_Y=2>
__global__ void matrixMultiplyV100(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE+1];

    float accum[THREAD_ITEMS_Y][THREAD_ITEMS_X] = {{0.0f}};

    const int row_base = by * BLOCK_SIZE + ty * THREAD_ITEMS_Y;
    const int col_base = bx * BLOCK_SIZE + tx * THREAD_ITEMS_X;

    const int TILE_ITEMS_X = BLOCK_SIZE / THREAD_ITEMS_X;
    const int TILE_ITEMS_Y = BLOCK_SIZE / THREAD_ITEMS_Y;

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {

        #pragma unroll
        for (int i = 0; i < THREAD_ITEMS_Y; ++i) {
            const int row = row_base + i;
            if (row < M) {
                const int a_col_base = tile * BLOCK_SIZE;
                #pragma unroll
                for (int j = 0; j < THREAD_ITEMS_X; ++j) {
                    const int a_col = a_col_base + tx * THREAD_ITEMS_X + j;
                    if (a_col < K) {
                        As[ty * THREAD_ITEMS_Y + i][tx * THREAD_ITEMS_X + j] = __ldg(&A[row * K + a_col]);
                    } else {
                        As[ty * THREAD_ITEMS_Y + i][tx * THREAD_ITEMS_X + j] = 0.0f;
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < THREAD_ITEMS_X; ++j) {
                    As[ty * THREAD_ITEMS_Y + i][tx * THREAD_ITEMS_X + j] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < THREAD_ITEMS_Y; ++i) {
            const int b_row = tile * BLOCK_SIZE + ty * THREAD_ITEMS_Y + i;
            if (b_row < K) {
                #pragma unroll
                for (int j = 0; j < THREAD_ITEMS_X; ++j) {
                    const int col = col_base + j;
                    if (col < N) {
                        Bs[ty * THREAD_ITEMS_Y + i][tx * THREAD_ITEMS_X + j] = __ldg(&B[b_row * N + col]);
                    } else {
                        Bs[ty * THREAD_ITEMS_Y + i][tx * THREAD_ITEMS_X + j] = 0.0f;
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < THREAD_ITEMS_X; ++j) {
                    Bs[ty * THREAD_ITEMS_Y + i][tx * THREAD_ITEMS_X + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < THREAD_ITEMS_Y; ++i) {
                const float a_val = As[ty * THREAD_ITEMS_Y + i][k];
                #pragma unroll
                for (int j = 0; j < THREAD_ITEMS_X; ++j) {
                    accum[i][j] += a_val * Bs[k][tx * THREAD_ITEMS_X + j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_ITEMS_Y; ++i) {
        const int row = row_base + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < THREAD_ITEMS_X; ++j) {
                const int col = col_base + j;
                if (col < N) {
                    C[row * N + col] = accum[i][j];
                }
            }
        }
    }
}

template<int BLOCK_SIZE>
__global__ void matrixMultiplyMixedPrecision(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE+1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    double sum = 0.0;

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {

        if (row < M && tile * BLOCK_SIZE + tx < K) {
            As[ty][tx] = __ldg(&A[row * K + (tile * BLOCK_SIZE + tx)]);
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((tile * BLOCK_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = __ldg(&B[(tile * BLOCK_SIZE + ty) * N + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum = __fma_rd(static_cast<double>(As[ty][k]),
                         static_cast<double>(Bs[k][tx]),
                         sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = static_cast<float>(sum);
    }
}

template<int BLOCK_SIZE>
__global__ void matrixMultiplyV100Large(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {

    constexpr int ITEMS_PER_THREAD_X = 4;
    constexpr int ITEMS_PER_THREAD_Y = 4;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE+4];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE+4];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row_base = by * BLOCK_SIZE + ty * ITEMS_PER_THREAD_Y;
    const int col_base = bx * BLOCK_SIZE + tx * ITEMS_PER_THREAD_X;

    float accum[ITEMS_PER_THREAD_Y][ITEMS_PER_THREAD_X] = {0.0f};

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {

        for (int i = 0; i < ITEMS_PER_THREAD_Y; ++i) {
            for (int j = 0; j < ITEMS_PER_THREAD_X; ++j) {
                int row = row_base + i;
                int col = tile * BLOCK_SIZE + tx * ITEMS_PER_THREAD_X + j;

                if (row < M && col < K) {
                    As[ty * ITEMS_PER_THREAD_Y + i][tx * ITEMS_PER_THREAD_X + j] = __ldg(&A[row * K + col]);
                } else {
                    As[ty * ITEMS_PER_THREAD_Y + i][tx * ITEMS_PER_THREAD_X + j] = 0.0f;
                }

                row = tile * BLOCK_SIZE + ty * ITEMS_PER_THREAD_Y + i;
                col = col_base + j;

                if (row < K && col < N) {
                    Bs[ty * ITEMS_PER_THREAD_Y + i][tx * ITEMS_PER_THREAD_X + j] = __ldg(&B[row * N + col]);
                } else {
                    Bs[ty * ITEMS_PER_THREAD_Y + i][tx * ITEMS_PER_THREAD_X + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD_Y; ++i) {
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD_X; ++j) {
                    accum[i][j] += As[ty * ITEMS_PER_THREAD_Y + i][k] *
                                  Bs[k][tx * ITEMS_PER_THREAD_X + j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < ITEMS_PER_THREAD_Y; ++i) {
        const int row = row_base + i;
        if (row < M) {
            for (int j = 0; j < ITEMS_PER_THREAD_X; ++j) {
                const int col = col_base + j;
                if (col < N) {
                    C[row * N + col] = accum[i][j];
                }
            }
        }
    }
}

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path,
                       int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "solution.dat";
        std::ofstream sol_file(sol_path, std::ios::binary);

        cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);

        float *h_A, *h_B, *h_C;
        CUDA_CHECK(cudaMallocHost(&h_A, n*k*sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_B, k*m*sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_C, n*m*sizeof(float)));

        std::ifstream(m1_path).read(reinterpret_cast<char*>(h_A), sizeof(float)*n*k);
        std::ifstream(m2_path).read(reinterpret_cast<char*>(h_B), sizeof(float)*k*m);

        bool useOptimized = true;

        CUDA_CHECK(cudaSetDevice(0));

        int maxThreadsPerMultiProcessor;
        int multiProcessorCount;
        CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor,
                                        cudaDevAttrMaxThreadsPerMultiProcessor, 0));
        CUDA_CHECK(cudaDeviceGetAttribute(&multiProcessorCount,
                                        cudaDevAttrMultiProcessorCount, 0));

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, n*k*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, k*m*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, n*m*sizeof(float)));

        constexpr int NUM_STREAMS = 4;
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        CUDA_CHECK(cudaStreamDestroy(streams[0]));

        int leastPriority, greatestPriority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams[0], cudaStreamNonBlocking, greatestPriority));

        CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, n*k*sizeof(float), cudaMemcpyHostToDevice, streams[1]));
        CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, k*m*sizeof(float), cudaMemcpyHostToDevice, streams[2]));

        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
        CUDA_CHECK(cudaStreamSynchronize(streams[2]));

        constexpr int BLOCK_SIZE = 64;

        if (useOptimized && n >= 2048 && m >= 2048 && k >= 2048) {

            constexpr int ITEMS_PER_THREAD = 4;

            dim3 threadsPerBlock(BLOCK_SIZE/ITEMS_PER_THREAD, BLOCK_SIZE/ITEMS_PER_THREAD);
            dim3 blocksPerGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            matrixMultiplyV100Large<BLOCK_SIZE>
                <<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(d_A, d_B, d_C, n, k, m);
        }
        else if (useOptimized && n >= 1024 && m >= 1024) {

            constexpr int ITEMS_PER_THREAD_X = 2;
            constexpr int ITEMS_PER_THREAD_Y = 2;

            dim3 threadsPerBlock(BLOCK_SIZE/ITEMS_PER_THREAD_X, BLOCK_SIZE/ITEMS_PER_THREAD_Y);
            dim3 blocksPerGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            matrixMultiplyV100<BLOCK_SIZE, ITEMS_PER_THREAD_X, ITEMS_PER_THREAD_Y>
                <<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(d_A, d_B, d_C, n, k, m);
        }
        else if (useOptimized) {

            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 blocksPerGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            matrixMultiplyMixedPrecision<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>
                (d_A, d_B, d_C, n, k, m);
        }
        else {

            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 blocksPerGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            matrixMultiplyTiled<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>
                (d_A, d_B, d_C, n, k, m);
        }

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, n*m*sizeof(float), cudaMemcpyDeviceToHost, streams[0]));

        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        sol_file.write(reinterpret_cast<const char*>(h_C), sizeof(float)*n*m);
        sol_file.close();

        CUDA_CHECK(cudaFreeHost(h_A));
        CUDA_CHECK(cudaFreeHost(h_B));
        CUDA_CHECK(cudaFreeHost(h_C));

        return sol_path;
    }
}