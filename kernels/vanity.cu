#include <stdio.h>
#include "base58.h"
#include "vanity.h"
#include "sha256.h"

__device__ int done = 0;
__device__ unsigned long long count = 0;

__device__ bool d_case_insensitive = false;

// TODO:
// 1) Should maybe write a macro for the err handling
// 2) Theoretically possible to reuse device reallocs but it's only one per round so it's kind of ok
extern "C" void vanity_round(
    int id,
    uint8_t *seed,
    uint8_t *base,
    uint8_t *owner,
    char *target,
    uint64_t target_len,
    uint8_t *out,
    bool case_insensitive)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (id >= deviceCount)
    {
        printf("Invalid GPU index: %d\n", id);
        return;
    }

    // Set device and initialize it
    cudaSetDevice(id);
    gpu_init(id);

    // Allocate device buffer for seed, base, owner, out, target len, target
    uint8_t *d_buffer;
    cudaError_t err = cudaMalloc(
        (void **)&d_buffer,
        32               // seed
            + 32         // base
            + 32         // owner
            + 8          // target len
            + target_len // target
            + 16         // out (16 byte seed)
    );
    if (err != cudaSuccess)
    {
        printf("CUDA malloc error (d_buffer): %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy input seed, base, owner to device
    err = cudaMemcpy(d_buffer, seed, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (seed): %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(d_buffer + 32, base, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (base): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    err = cudaMemcpy(d_buffer + 64, owner, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (owner): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    err = cudaMemcpy(d_buffer + 96, &target_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (target_len): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    err = cudaMemcpy(d_buffer + 104, target, target_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (target): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    err = cudaMemcpyToSymbol(d_case_insensitive, &case_insensitive, 1, 0, cudaMemcpyHostToDevice);

    // Reset tracker and counter using cudaMemcpyToSymbol
    int zero = 0;
    unsigned long long zero_ull = 0;
    err = cudaMemcpyToSymbol(done, &zero, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy to symbol error (done): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    err = cudaMemcpyToSymbol(count, &zero_ull, sizeof(unsigned long long));
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy to symbol error (count): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    // Launch vanity search kernel
    vanity_search<<<num_blocks, num_threads>>>(d_buffer, num_blocks * num_threads);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    // Copy result to host
    err = cudaMemcpy(out, d_buffer + 104 + target_len, 16, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (d_out): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    err = cudaMemcpyFromSymbol(out + 16, count, 8, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    // Free pointers
    cudaFree(d_buffer);
}

__device__ uint8_t const alphanumeric[63] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

__global__ void
vanity_search(uint8_t *buffer, uint64_t stride)
{
    // Deconstruct buffer
    uint8_t *seed = buffer;
    uint8_t *base = buffer + 32;
    uint8_t *owner = buffer + 64;
    uint64_t target_len;
    memcpy(&target_len, buffer + 96, 8);
    uint8_t *target = buffer + 104;
    uint8_t *out = (buffer + 104 + target_len);

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char local_out[32] = {0};
    unsigned char local_encoded[44] = {0};
    uint64_t local_seed[4];

    // Pseudo random generator
    CUDA_SHA256_CTX ctx;
    cuda_sha256_init(&ctx);
    cuda_sha256_update(&ctx, (BYTE *)(seed), 32);
    cuda_sha256_update(&ctx, (BYTE *)(&idx), 8);
    cuda_sha256_final(&ctx, (BYTE *)local_seed);

    CUDA_SHA256_CTX address_sha;
    cuda_sha256_init(&address_sha);
    cuda_sha256_update(&address_sha, (BYTE *)base, 32);

    for (uint64_t iter = 0; iter < 1000 * 1000 * 1000; iter++)
    {
        // Has someone found a result?
        if (iter % 100 == 0)
        {
            if (atomicMax(&done, 0) == 1)
            {
                atomicAdd(&count, iter);
                return;
            }
        }

        cuda_sha256_init(&ctx);
        cuda_sha256_update(&ctx, (BYTE *)local_seed, 16);
        cuda_sha256_final(&ctx, (BYTE *)local_seed);

        uint32_t *indices = (uint32_t *)&local_seed;
        uint8_t create_account_seed[16] = {
            alphanumeric[indices[0] % 62],
            alphanumeric[indices[1] % 62],
            alphanumeric[indices[2] % 62],
            alphanumeric[indices[3] % 62],
            alphanumeric[indices[4] % 62],
            alphanumeric[indices[5] % 62],
            alphanumeric[indices[6] % 62],
            alphanumeric[indices[7] % 62],
            alphanumeric[(indices[0] >> 2) % 62],
            alphanumeric[(indices[1] >> 2) % 62],
            alphanumeric[(indices[2] >> 2) % 62],
            alphanumeric[(indices[3] >> 2) % 62],
            alphanumeric[(indices[4] >> 2) % 62],
            alphanumeric[(indices[5] >> 2) % 62],
            alphanumeric[(indices[6] >> 2) % 62],
            alphanumeric[(indices[7] >> 2) % 62],
        };

        // Calculate and encode public
        CUDA_SHA256_CTX address_sha_local;
        memcpy(&address_sha_local, &address_sha, sizeof(CUDA_SHA256_CTX));
        cuda_sha256_update(&address_sha_local, (BYTE *)create_account_seed, 16);
        cuda_sha256_update(&address_sha_local, (BYTE *)owner, 32);
        cuda_sha256_final(&address_sha_local, (BYTE *)local_out);
        fd_base58_encode_32(local_out, (unsigned char *)(&local_encoded), d_case_insensitive);

        // Check target
        if (matches_target((unsigned char *)local_encoded, (unsigned char *)target, target_len))
        {
            // Are we first to write result?
            if (atomicMax(&done, 1) == 0)
            {
                // seed for CreateAccountWithSeed
                memcpy(out, create_account_seed, 16);
            }

            atomicAdd(&count, iter + 1);
            return;
        }
    }
}

__device__ bool matches_target(unsigned char *a, unsigned char *target, uint64_t n)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != target[i])
            return false;
    }
    return true;
}

__device__ uint8_t hi[32] = {
    11, 173, 87, 184, 147, 234, 104, 132, 137, 81, 39, 28, 37, 108, 223, 35, 44, 157, 207, 159, 1, 96, 51, 104, 99, 171, 194, 255, 255, 255, 255, 255};
__device__ uint8_t lo[32] = {
    11, 173, 87, 184, 50, 176, 108, 7, 110, 59, 244, 93, 50, 24, 226, 112, 40, 115, 18, 145, 61, 45, 152, 223, 115, 133, 252, 192, 0, 0, 0, 0};
__device__ bool in_target_range(unsigned char *a)
{
    for (int i = 0; i < 32; i++)
    {
        if (a[i] == hi[i] && a[i] == lo[i])
            continue;
        if (a[i] <= hi[i] && a[i] >= lo[i])
            return true;
        else
            return false;
    }
    return false;
}
