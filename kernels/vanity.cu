#include <stdio.h>
#include "base58.h"
#include "vanity.h"
#include "sha256.h"

__device__ int done = 0;
__device__ unsigned long long count = 0;
__device__ bool d_case_insensitive_prefix = false;
__device__ bool d_case_insensitive_suffix = false;
__device__ bool d_case_insensitive_any = false;
__device__ bool d_leet_speak = false;

__device__ char device_tolower(char c)
{
    if (c >= 'A' && c <= 'Z')
    {
        return c + ('a' - 'A');
    }
    return c;
}

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
    char *any,
    uint64_t any_len,
    uint8_t *out,
    bool case_insensitive_prefix,
    bool case_insensitive_suffix,
    bool case_insensitive_any,
    bool leet_speak)
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

    // Calculate total buffer size with padding for alignment
    size_t total_buffer_size =
        32 +         // seed
        32 +         // base
        32 +         // owner
        8 +          // prefix len
        prefix_len + // prefix
        8 +          // suffix len
        suffix_len + // suffix
        8 +          // any len
        any_len +    // any string
        16 +         // out (16 byte seed)
        8;           // padding for alignment

    // Ensure buffer size is aligned to 16 bytes
    total_buffer_size = (total_buffer_size + 15) & ~15;

    printf("Allocating aligned buffer of size: %zu bytes\n", total_buffer_size);

    // Allocate device buffer
    uint8_t *d_buffer;
    cudaError_t err = cudaMalloc((void **)&d_buffer, total_buffer_size);
    if (err != cudaSuccess)
    {
        printf("CUDA malloc error (d_buffer): %s\n", cudaGetErrorString(err));
        return;
    }

    // Initialize buffer to zero
    err = cudaMemset(d_buffer, 0, total_buffer_size);
    if (err != cudaSuccess)
    {
        printf("CUDA memset error: %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
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

    // Copy any length and string
    err = cudaMemcpy(d_buffer + 112 + prefix_len + suffix_len, &any_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy error (any_len): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    if (any_len > 0)
    {
        err = cudaMemcpy(d_buffer + 120 + prefix_len + suffix_len, any, any_len, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            printf("CUDA memcpy error (any): %s\n", cudaGetErrorString(err));
            cudaFree(d_buffer);
            return;
        }
    }

    err = cudaMemcpyToSymbol(d_case_insensitive_prefix, &case_insensitive_prefix, 1, 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy to symbol error (case_insensitive_prefix): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    err = cudaMemcpyToSymbol(d_case_insensitive_suffix, &case_insensitive_suffix, 1, 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy to symbol error (case_insensitive_suffix): %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    err = cudaMemcpyToSymbol(d_case_insensitive_any, &case_insensitive_any, 1, 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy to symbol error (case_insensitive_any): %s\n", cudaGetErrorString(err));
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

    // Copy leet_speak setting to device
    err = cudaMemcpyToSymbol(d_leet_speak, &leet_speak, sizeof(bool));
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy to symbol error (leet_speak): %s\n", cudaGetErrorString(err));
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

    // Fix output copy location
    size_t out_offset = 120 + prefix_len + suffix_len + any_len;
    err = cudaMemcpy(out, d_buffer + out_offset, 16, cudaMemcpyDeviceToHost);
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
    // Get thread index
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stride)
    {
        return;
    }

    // Deconstruct buffer with bounds checking
    uint64_t prefix_len = 0;
    uint64_t suffix_len = 0;
    uint64_t any_len = 0;

    // Read lengths with bounds checking
    if (memcpy(&prefix_len, buffer + 96, 8) != NULL &&
        prefix_len <= 44 &&
        memcpy(&suffix_len, buffer + 104 + prefix_len, 8) != NULL &&
        suffix_len <= 44 &&
        memcpy(&any_len, buffer + 112 + prefix_len + suffix_len, 8) != NULL &&
        any_len <= 44)
    {
        // Get the pointers to the strings
        uint8_t *prefix = buffer + 104;
        uint8_t *suffix = buffer + 112 + prefix_len;
        uint8_t *any = buffer + 120 + prefix_len + suffix_len;
        uint8_t *out = buffer + 120 + prefix_len + suffix_len + any_len;

        // Add size verification
        if (prefix_len > 44 || suffix_len > 44 || any_len > 44)
        {
            printf("Invalid string length: prefix=%lu, suffix=%lu, any=%lu\n",
                   prefix_len, suffix_len, any_len);
            return;
        }

        unsigned char local_out[32] = {0};
        unsigned char local_encoded[44] = {0};
        uint64_t local_seed[4];

        // Pseudo random generator
        CUDA_SHA256_CTX ctx;
        cuda_sha256_init(&ctx);
        cuda_sha256_update(&ctx, (BYTE *)(buffer), 32);
        cuda_sha256_update(&ctx, (BYTE *)(&idx), 8);
        cuda_sha256_final(&ctx, (BYTE *)local_seed);

        CUDA_SHA256_CTX address_sha;
        cuda_sha256_init(&address_sha);
        cuda_sha256_update(&address_sha, (BYTE *)(buffer + 32), 32);

        // Pre-initialize the context for seed updates
        CUDA_SHA256_CTX seed_update_ctx;
        cuda_sha256_init(&seed_update_ctx);

        for (uint64_t iter = 0; iter < 1000 * 1000 * 1000; iter++)
        {
            // Has someone found a result?
            if (iter % 1000 == 0)
            {
                if (atomicMax(&done, 0) == 1)
                {
                    atomicAdd(&count, iter);
                    return;
                }
            }

            // Reuse pre-initialized context
            memcpy(&ctx, &seed_update_ctx, sizeof(CUDA_SHA256_CTX));
            cuda_sha256_update(&ctx, (BYTE *)local_seed, 16);
            cuda_sha256_final(&ctx, (BYTE *)local_seed);

            uint32_t *indices = (uint32_t *)&local_seed;
            // Cache frequently used values
            const uint32_t idx0 = indices[0];
            const uint32_t idx1 = indices[1];
            const uint32_t idx2 = indices[2];
            const uint32_t idx3 = indices[3];
            const uint32_t idx4 = indices[4];
            const uint32_t idx5 = indices[5];
            const uint32_t idx6 = indices[6];
            const uint32_t idx7 = indices[7];

            uint8_t create_account_seed[16] = {
                alphanumeric[idx0 % 62],
                alphanumeric[idx1 % 62],
                alphanumeric[idx2 % 62],
                alphanumeric[idx3 % 62],
                alphanumeric[idx4 % 62],
                alphanumeric[idx5 % 62],
                alphanumeric[idx6 % 62],
                alphanumeric[idx7 % 62],
                alphanumeric[(idx0 >> 2) % 62],
                alphanumeric[(idx1 >> 2) % 62],
                alphanumeric[(idx2 >> 2) % 62],
                alphanumeric[(idx3 >> 2) % 62],
                alphanumeric[(idx4 >> 2) % 62],
                alphanumeric[(idx5 >> 2) % 62],
                alphanumeric[(idx6 >> 2) % 62],
                alphanumeric[(idx7 >> 2) % 62],
            };

            // Calculate and encode public
            CUDA_SHA256_CTX address_sha_local;
            memcpy(&address_sha_local, &address_sha, sizeof(CUDA_SHA256_CTX));
            cuda_sha256_update(&address_sha_local, (BYTE *)create_account_seed, 16);
            cuda_sha256_update(&address_sha_local, (BYTE *)(buffer + 64), 32);
            cuda_sha256_final(&address_sha_local, (BYTE *)local_out);
            fd_base58_encode_32(local_out, (unsigned char *)(&local_encoded), d_case_insensitive_prefix || d_case_insensitive_suffix || d_case_insensitive_any);

            // Check prefix and suffix
            // printf("Got key: %s\n", local_encoded);

            if (matches_search(
                    (unsigned char *)local_encoded,
                    (unsigned char *)prefix,
                    prefix_len,
                    (unsigned char *)suffix,
                    suffix_len,
                    (unsigned char *)any,
                    any_len))
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
    else
    {
        printf("Error: Invalid buffer lengths or memory access\n");
        return;
    }
}

__device__ bool chars_match_leet(char a, char b)
{
    if (a == b)
        return true;

    switch (a)
    {
    case 'a':
    case 'A':
        return b == '4';
    case 'e':
    case 'E':
        return b == '3';
    case 't':
    case 'T':
        return b == '7';
    case 'l':
    case 'L':
        return b == '1';
    case 'i':
    case 'I':
        return b == '1';
    case 's':
    case 'S':
        return b == '5';
    case 'g':
    case 'G':
        return b == '6';
    case 'b':
    case 'B':
        return b == '8';
    case 'z':
    case 'Z':
        return b == '2';
    }

    switch (b)
    {
    case '4':
        return a == 'a' || a == 'A';
    case '3':
        return a == 'e' || a == 'E';
    case '7':
        return a == 't' || a == 'T';
    case '1':
        return a == 'l' || a == 'L' || a == 'i' || a == 'I';
    case '5':
        return a == 's' || a == 'S';
    case '6':
        return a == 'g' || a == 'G';
    case '8':
        return a == 'b' || a == 'B';
    case '2':
        return a == 'z' || a == 'Z';
    }

    return false;
}

__device__ bool matches_search(
    unsigned char *address,
    unsigned char *prefix,
    uint64_t prefix_len,
    unsigned char *suffix,
    uint64_t suffix_len,
    unsigned char *any,
    uint64_t any_len)
{
    bool prefix_matches = true;
    bool suffix_matches = true;
    bool any_matches = true;

    // Skip checks if length is 0
    if (prefix_len > 0)
    {
        // Check prefix
        for (int i = 0; i < prefix_len; i++)
        {
            if (d_leet_speak)
            {
                if (!chars_match_leet(prefix[i], address[i]))
                {
                    prefix_matches = false;
                    break;
                }
            }
            else if (d_case_insensitive_prefix)
            {
                if (device_tolower(address[i]) != device_tolower(prefix[i]))
                {
                    prefix_matches = false;
                    break;
                }
            }
            else if (address[i] != prefix[i])
            {
                prefix_matches = false;
                break;
            }
        }
    }

    if (suffix_len > 0)
    {
        // Check suffix
        for (int i = 0; i < suffix_len; i++)
        {
            if (d_leet_speak)
            {
                if (!chars_match_leet(suffix[i], address[44 - suffix_len + i]))
                {
                    suffix_matches = false;
                    break;
                }
            }
            else if (d_case_insensitive_suffix)
            {
                if (device_tolower(address[44 - suffix_len + i]) != device_tolower(suffix[i]))
                {
                    suffix_matches = false;
                    break;
                }
            }
            else if (address[44 - suffix_len + i] != suffix[i])
            {
                suffix_matches = false;
                break;
            }
        }
    }

    if (any_len > 0)
    {
        // Check for 'any' string anywhere in the address
        any_matches = false;
        for (int i = 0; i <= 44 - any_len; i++)
        {
            bool match = true;
            for (int j = 0; j < any_len; j++)
            {
                if (d_leet_speak)
                {
                    if (!chars_match_leet(any[j], address[i + j]))
                    {
                        match = false;
                        break;
                    }
                }
                else if (d_case_insensitive_any)
                {
                    if (device_tolower(address[i + j]) != device_tolower(any[j]))
                    {
                        match = false;
                        break;
                    }
                }
                else if (address[i + j] != any[j])
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                any_matches = true;
                break;
            }
        }
    }

    bool final_match = prefix_matches && suffix_matches && any_matches;

    // Only print if we found a match, with bounds checking
    if (final_match)
    {
        printf("\nCUDA MATCH FOUND!\n");
        printf("Full address: %.44s\n", (char *)address);
        printf("Raw seed bytes: [");
        for (int i = 0; i < 16; i++)
        {
            printf("%02x", ((uint8_t *)out)[i]);
            if (i < 15)
                printf(", ");
        }
        printf("]\n");
        printf("Raw seed UTF-8: ");
        for (int i = 0; i < 16; i++)
        {
            printf("%c", ((char *)out)[i]);
        }
        printf("\n");

        if (prefix_len > 0 && prefix_len <= 44)
        {
            printf("Prefix match (%lu chars): '%.44s' - %s\n",
                   prefix_len, (char *)prefix, prefix_matches ? "YES" : "NO");
        }

        if (suffix_len > 0 && suffix_len <= 44)
        {
            printf("Suffix match (%lu chars): '%.44s' - %s\n",
                   suffix_len, (char *)suffix, suffix_matches ? "YES" : "NO");
        }

        if (any_len > 0 && any_len <= 44)
        {
            printf("Any match (%lu chars): '%.44s' - %s\n",
                   any_len, (char *)any, any_matches ? "YES" : "NO");
        }

        printf("Leet speak: %s\n", d_leet_speak ? "enabled" : "disabled");
        printf("Case insensitive prefix: %s\n", d_case_insensitive_prefix ? "enabled" : "disabled");
        printf("Case insensitive suffix: %s\n", d_case_insensitive_suffix ? "enabled" : "disabled");
        printf("Case insensitive any: %s\n", d_case_insensitive_any ? "enabled" : "disabled");
    }

    return final_match;
}
