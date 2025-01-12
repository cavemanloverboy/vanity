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
    char *prefix,
    uint64_t prefix_len,
    char *suffix,
    uint64_t suffix_len,
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


    // Allocate device buffer for seed, base, owner, out, prefix len, prefix, suffix_len, suffix
    uint8_t *d_buffer;
    cudaError_t err = cudaMalloc(
        (void **)&d_buffer,
        32               // seed
            + 32         // base
            + 32         // owner
            + 8          // prefix len
            + prefix_len // prefix
            + 8          // suffix len
            + suffix_len // suffix
            + 16         // out (16 byte seed)
    );
    printf("CUDA device count: %d\n", deviceCount);
    printf("Setting GPU device %d\n", id);
    printf("CUDA malloc successful for d_buffer\n");

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
    err = cudaMemcpy(d_buffer + 96, &prefix_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (prefix_len): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    
    // Copy prefix to device memory
    err = cudaMemcpy(d_buffer + 104, prefix, prefix_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (prefix): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    
    // Copy suffix length to device memory
    err = cudaMemcpy(d_buffer + 104 + prefix_len, &suffix_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (suffix_len): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    
    // Copy suffix to device memory
    err = cudaMemcpy(d_buffer + 112 + prefix_len, suffix, suffix_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (suffix): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    err = cudaMemcpyToSymbol(d_case_insensitive, &case_insensitive, 1, 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy to symbol error (done): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }
    
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

    printf("Launching vanity_search kernel\n");
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

    printf("Vanity search kernel launched and synchronized\n");

    // Copy result to host
    // Copy the output from device to host
    err = cudaMemcpy(out, d_buffer + 104 + prefix_len + suffix_len, 16, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (d_out): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    // Copy the 'count' value from the device to host
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
    uint64_t prefix_len;
    uint64_t suffix_len;
    
    // Assuming the prefix and suffix lengths are already in the buffer
    memcpy(&prefix_len, buffer + 96, 8);
    memcpy(&suffix_len, buffer + 104 + prefix_len, 8);  // Assuming suffix_len is placed after the prefix
    
    uint8_t *prefix = buffer + 104;  // The prefix starts after the prefix_len
    uint8_t *suffix = buffer + 104 + prefix_len;  // The suffix starts after the prefix data
    uint8_t *out = (buffer + 104 + prefix_len + suffix_len);  // Out is after both prefix and suffix
    

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

        // Check prefix and suffix
        // printf("Got key: %s\n", local_encoded);

        if (matches_search((unsigned char *)local_encoded, (unsigned char *)prefix, prefix_len, (unsigned char *)suffix, suffix_len))
        {
            // Are we first to write result?
            if (atomicMax(&done, 1) == 0)
            {
                // seed for CreateAccountWithSeed
                // printf("Match found! Copying result to out\n");

                memcpy(out, create_account_seed, 16);
            }

            atomicAdd(&count, iter + 1);
            return;
        }
    }
}

__device__ bool matches_search(unsigned char *a, unsigned char *prefix, uint64_t prefix_len, unsigned char *suffix, uint64_t suffix_len)
{
    for (int i = 0; i < prefix_len; i++) 
    {
        if (a[i] != prefix[i])
            return false;
    }

    for (int i = 0; i < suffix_len; i++) 
    {
        if (a[prefix_len + i] != suffix[i])
            return false;
    }

    return true;
}
