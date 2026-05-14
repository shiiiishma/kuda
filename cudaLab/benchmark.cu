#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// ============================
// CONSTANTS
// ============================

const int R = 100;
const int ANGLE_STEPS = 36;
const float PI = 3.14159265358979323846f;

// ============================
// GLOBAL MEMORY VERSION
// ============================

__global__ void detect_global(int* P, int n, int* K1, int* K2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;

    if (idx >= total) return;

    int x = idx / n;
    int y = idx % n;

    int val = P[idx];

    if (val == 3)
    {
        for (int t = 0; t < ANGLE_STEPS; t++)
        {
            float theta = 2.0f * PI * t / ANGLE_STEPS;

            int cx = x - (int)(R * cosf(theta));
            int cy = y - (int)(R * sinf(theta));

            if (cx >= 0 && cx < n && cy >= 0 && cy < n)
            {
                atomicAdd(K2, 1);
            }
        }
    }

    if (val == 2)
    {
        atomicAdd(K1, 1);
    }
}

// ============================
// SHARED MEMORY VERSION
// ============================

__global__ void detect_shared(int* P, int n, int* K1, int* K2)
{
    __shared__ int local_K1;
    __shared__ int local_K2;

    if (threadIdx.x == 0)
    {
        local_K1 = 0;
        local_K2 = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;

    if (idx < total)
    {
        int x = idx / n;
        int y = idx % n;

        int val = P[idx];

        if (val == 3)
        {
            for (int t = 0; t < ANGLE_STEPS; t++)
            {
                float theta = 2.0f * PI * t / ANGLE_STEPS;

                int cx = x - (int)(R * cosf(theta));
                int cy = y - (int)(R * sinf(theta));

                if (cx >= 0 && cx < n && cy >= 0 && cy < n)
                {
                    atomicAdd(&local_K2, 1);
                }
            }
        }

        if (val == 2)
        {
            atomicAdd(&local_K1, 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(K1, local_K1);
        atomicAdd(K2, local_K2);
    }
}

// ============================
// DATA GENERATION
// ============================

void generate(std::vector<int>& P, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        int r = rand() % 1000;

        if (r < 5) P[i] = 2;
        else if (r < 10) P[i] = 3;
        else P[i] = 0;
    }
}

// ============================
// EXPERIMENT
// ============================

void run_experiment(int n, int blocks, int threads)
{
    int size = n * n * sizeof(int);

    std::vector<int> h_P(n * n);
    generate(h_P, n);

    int *d_P, *d_K1, *d_K2;

    CHECK(cudaMalloc(&d_P, size));
    CHECK(cudaMalloc(&d_K1, sizeof(int)));
    CHECK(cudaMalloc(&d_K2, sizeof(int)));

    CHECK(cudaMemcpy(d_P, h_P.data(), size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_global = 0.0f;
    float total_shared = 0.0f;

    // ================= GLOBAL =================
    for (int i = 0; i < 100; i++)
    {
        CHECK(cudaMemset(d_K1, 0, sizeof(int)));
        CHECK(cudaMemset(d_K2, 0, sizeof(int)));

        CHECK(cudaEventRecord(start));

        detect_global<<<blocks, threads>>>(d_P, n, d_K1, d_K2);

        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_global += ms;
    }

    // ================= SHARED =================
    for (int i = 0; i < 100; i++)
    {
        CHECK(cudaMemset(d_K1, 0, sizeof(int)));
        CHECK(cudaMemset(d_K2, 0, sizeof(int)));

        CHECK(cudaEventRecord(start));

        detect_shared<<<blocks, threads>>>(d_P, n, d_K1, d_K2);

        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_shared += ms;
    }

    // ================= OUTPUT (СТАБИЛЬНЫЙ ФОРМАТ) =================

    float g = total_global / 100.0f;
    float s = total_shared / 100.0f;

    std::cout << "RESULT GLOBAL " << g << "\n";
    std::cout << "RESULT SHARED " << s << "\n";

    // debug info
    std::cout << "DEBUG N=" << n
              << " B=" << blocks
              << " T=" << threads << "\n";

    cudaFree(d_P);
    cudaFree(d_K1);
    cudaFree(d_K2);
}

// ============================
// MAIN
// ============================

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: ./hough_cuda N BLOCKS THREADS\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int blocks = atoi(argv[2]);
    int threads = atoi(argv[3]);

    run_experiment(n, blocks, threads);

    return 0;
}