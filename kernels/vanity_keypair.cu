#include <stdio.h>
#include "vanity_keypair.h"
#include "base58.h"
#include "sha256.h"

#include "ed25519/fe.cu"
#include "ed25519/ge.cu"
#include "ed25519/sha512.cu"

__device__ static int kp_done = 0;
__device__ static unsigned long long kp_count = 0;
__device__ static bool kp_case_insensitive = false;

static int kp_num_blocks;
static int kp_num_threads;

static void kp_gpu_init(int id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, id);

    kp_num_threads = 128;
    int blocks_per_sm = prop.maxThreadsPerMultiProcessor / kp_num_threads;
    kp_num_blocks = blocks_per_sm * prop.multiProcessorCount;
}

#define KP_MAX_THREADS 128

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_keypair_search(uint8_t *buffer, uint64_t stride);
static __device__ bool kp_matches_target(
    unsigned char *a,
    unsigned char *prefix, uint64_t prefix_len,
    unsigned char *suffix, uint64_t suffix_len,
    ulong encoded_len);

extern "C" void vanity_keypair_round(
    int id,
    uint8_t *seed,
    char *prefix,
    char *suffix,
    uint64_t prefix_len,
    uint64_t suffix_len,
    uint8_t *out,
    bool case_insensitive)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (id >= deviceCount) {
        printf("Invalid GPU index: %d\n", id);
        return;
    }

    cudaSetDevice(id);
    kp_gpu_init(id);

    uint8_t *d_buffer;
    uint64_t buf_size =
        32              // seed
        + 8             // prefix_len
        + prefix_len    // prefix
        + 8             // suffix_len
        + suffix_len    // suffix
        + 32            // out seed
        ;

    cudaError_t err = cudaMalloc((void **)&d_buffer, buf_size);
    if (err != cudaSuccess) {
        printf("CUDA malloc error: %s\n", cudaGetErrorString(err));
        return;
    }

    uint64_t off = 0;

    err = cudaMemcpy(d_buffer + off, seed, 32, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (seed): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }
    off += 32;

    err = cudaMemcpy(d_buffer + off, &prefix_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (prefix_len): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }
    off += 8;

    err = cudaMemcpy(d_buffer + off, prefix, prefix_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (prefix): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }
    off += prefix_len;

    err = cudaMemcpy(d_buffer + off, &suffix_len, 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (suffix_len): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }
    off += 8;

    err = cudaMemcpy(d_buffer + off, suffix, suffix_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy error (suffix): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }
    off += suffix_len;

    err = cudaMemcpyToSymbol(kp_case_insensitive, &case_insensitive, 1, 0, cudaMemcpyHostToDevice);

    int zero = 0;
    unsigned long long zero_ull = 0;
    cudaMemcpyToSymbol(kp_done, &zero, sizeof(int));
    cudaMemcpyToSymbol(kp_count, &zero_ull, sizeof(unsigned long long));

    vanity_keypair_search<<<kp_num_blocks, kp_num_threads>>>(d_buffer, kp_num_blocks * kp_num_threads);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_buffer);
        return;
    }

    // out layout: 32 bytes winning seed + 8 bytes count
    err = cudaMemcpy(out, d_buffer + off, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (out seed): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    err = cudaMemcpyFromSymbol(out + 32, kp_count, 8, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA memcpy error (count): %s\n", cudaGetErrorString(err)); cudaFree(d_buffer); return; }

    cudaFree(d_buffer);
}

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_keypair_search(uint8_t *buffer, uint64_t stride)
{
    uint8_t *host_seed = buffer;

    uint64_t prefix_len;
    memcpy(&prefix_len, buffer + 32, 8);
    uint8_t *prefix = buffer + 40;

    uint64_t suffix_len;
    memcpy(&suffix_len, buffer + 40 + prefix_len, 8);
    uint8_t *suffix = buffer + 40 + prefix_len + 8;

    uint8_t *out = suffix + suffix_len;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char seed[32];
    unsigned char privatek[64];
    unsigned char pubkey[32];
    unsigned char encoded[45];
    ge_p3 A;

    // Derive unique per-thread starting seed: SHA-256(host_seed || thread_idx)
    CUDA_SHA256_CTX sha256_ctx;
    cuda_sha256_init(&sha256_ctx);
    cuda_sha256_update(&sha256_ctx, (BYTE *)host_seed, 32);
    cuda_sha256_update(&sha256_ctx, (BYTE *)(&idx), 8);
    cuda_sha256_final(&sha256_ctx, (BYTE *)seed);

    for (uint64_t iter = 0; iter < uint64_t(1000) * 1000 * 1000 * 1000; iter++)
    {
        if (iter % 100 == 0) {
            if (atomicMax(&kp_done, 0) == 1) {
                atomicAdd(&kp_count, iter);
                return;
            }
        }

        // Inlined SHA-512 for exactly 32 bytes:
        //   sha512_init -> copy 32 bytes into buf -> sha512_final
        // This avoids all branching in sha512_update since input < 128 bytes.
        sha512_context md;
        md.curlen = 0;
        md.length = 0;
        md.state[0] = UINT64_C(0x6a09e667f3bcc908);
        md.state[1] = UINT64_C(0xbb67ae8584caa73b);
        md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
        md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
        md.state[4] = UINT64_C(0x510e527fade682d1);
        md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
        md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
        md.state[7] = UINT64_C(0x5be0cd19137e2179);

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            md.buf[i] = seed[i];
        }
        md.curlen = 32;

        sha512_final(&md, privatek);

        // ed25519 clamping
        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(pubkey, &A);

        ulong enc_len = fd_base58_encode_32(pubkey, encoded, kp_case_insensitive);

        if (kp_matches_target(encoded, prefix, prefix_len, suffix, suffix_len, enc_len))
        {
            if (atomicMax(&kp_done, 1) == 0) {
                memcpy(out, seed, 32);
            }
            atomicAdd(&kp_count, iter + 1);
            return;
        }

        // Zero-cost RNG: second half of SHA-512 output becomes next seed
        memcpy(seed, privatek + 32, 32);
    }
}

static __device__ bool kp_matches_target(
    unsigned char *a,
    unsigned char *prefix, uint64_t prefix_len,
    unsigned char *suffix, uint64_t suffix_len,
    ulong encoded_len)
{
    for (uint64_t i = 0; i < prefix_len; i++) {
        if (a[i] != prefix[i])
            return false;
    }
    for (uint64_t i = 0; i < suffix_len; i++) {
        if (a[encoded_len - suffix_len + i] != suffix[i])
            return false;
    }
    return true;
}
