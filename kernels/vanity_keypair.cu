#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "vanity_keypair.h"
#include "base58.h"
#include "sha256.h"

#include "ed25519/fe.cu"
#include "ed25519/ge.cu"
#include "ed25519/sha512.cu"

__device__ static int kp_done = 0;
__device__ static unsigned long long kp_count = 0;
__device__ static bool kp_case_insensitive = false;

#define KP_MAX_THREADS 128

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_keypair_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles);
static __device__ bool kp_matches_target(
    unsigned char *a,
    unsigned char *prefix, uint64_t prefix_len,
    unsigned char *suffix, uint64_t suffix_len,
    ulong encoded_len);

// ─── persistent context ─────────────────────────────────────────────────────

typedef struct {
    int device_id;
    cudaStream_t stream;
    uint8_t *d_buffer;
    int num_blocks;
    int num_threads;
    unsigned long long target_cycles;
    uint64_t out_offset;
} GpuKeypairCtx;

extern "C" void* gpu_keypair_init(
    int id,
    uint8_t *prefix, uint64_t prefix_len,
    uint8_t *suffix, uint64_t suffix_len,
    bool case_insensitive)
{
    cudaSetDevice(id);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, id);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_keypair_init(%d): cudaGetDeviceProperties: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int nthreads = KP_MAX_THREADS;
    int blocks_per_sm = prop.maxThreadsPerMultiProcessor / nthreads;
    int nblocks = blocks_per_sm * prop.multiProcessorCount;

    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, id) != cudaSuccess || clock_khz <= 0) {
        fprintf(stderr, "gpu_keypair_init(%d): clock rate query failed\n", id);
        exit(EXIT_FAILURE);
    }

    GpuKeypairCtx *ctx = (GpuKeypairCtx *)malloc(sizeof(GpuKeypairCtx));
    ctx->device_id   = id;
    ctx->num_blocks  = nblocks;
    ctx->num_threads = nthreads;
    ctx->target_cycles = (unsigned long long)clock_khz * 1000ULL * 5ULL;

    cudaStreamCreate(&ctx->stream);

    // Buffer: [seed:32] [prefix_len:8] [prefix:N] [suffix_len:8] [suffix:M] [out:32]
    uint64_t buf_size = 32 + 8 + prefix_len + 8 + suffix_len + 32;
    ctx->out_offset = 32 + 8 + prefix_len + 8 + suffix_len;

    err = cudaMalloc((void**)&ctx->d_buffer, buf_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_keypair_init(%d): cudaMalloc: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Upload invariant data (everything after the 32-byte seed slot)
    uint64_t off = 32;
    cudaMemcpy(ctx->d_buffer + off, &prefix_len, 8, cudaMemcpyHostToDevice); off += 8;
    if (prefix_len > 0) cudaMemcpy(ctx->d_buffer + off, prefix, prefix_len, cudaMemcpyHostToDevice);
    off += prefix_len;
    cudaMemcpy(ctx->d_buffer + off, &suffix_len, 8, cudaMemcpyHostToDevice); off += 8;
    if (suffix_len > 0) cudaMemcpy(ctx->d_buffer + off, suffix, suffix_len, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(kp_case_insensitive, &case_insensitive, 1, 0, cudaMemcpyHostToDevice);

    return (void *)ctx;
}

extern "C" void gpu_keypair_launch(void *opaque, uint8_t *seed)
{
    GpuKeypairCtx *ctx = (GpuKeypairCtx *)opaque;
    cudaSetDevice(ctx->device_id);

    cudaMemcpy(ctx->d_buffer, seed, 32, cudaMemcpyHostToDevice);

    int zero = 0;
    unsigned long long zero_ull = 0;
    cudaMemcpyToSymbol(kp_done, &zero, sizeof(int));
    cudaMemcpyToSymbol(kp_count, &zero_ull, sizeof(unsigned long long));

    vanity_keypair_search<<<ctx->num_blocks, ctx->num_threads, 0, ctx->stream>>>(
        ctx->d_buffer,
        (uint64_t)ctx->num_blocks * ctx->num_threads,
        ctx->target_cycles);
}

extern "C" int gpu_keypair_query(void *opaque)
{
    GpuKeypairCtx *ctx = (GpuKeypairCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    return cudaStreamQuery(ctx->stream) == cudaSuccess ? 1 : 0;
}

// out layout written by caller: [seed:32] [count:8]
extern "C" void gpu_keypair_read(void *opaque, uint8_t *out)
{
    GpuKeypairCtx *ctx = (GpuKeypairCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    cudaMemcpy(out, ctx->d_buffer + ctx->out_offset, 32, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(out + 32, kp_count, 8, 0, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_keypair_destroy(void *opaque)
{
    GpuKeypairCtx *ctx = (GpuKeypairCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    cudaStreamSynchronize(ctx->stream);
    cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_buffer);
    free(ctx);
}

// ─── kernel (unchanged) ─────────────────────────────────────────────────────

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_keypair_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles)
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

    CUDA_SHA256_CTX sha256_ctx;
    cuda_sha256_init(&sha256_ctx);
    cuda_sha256_update(&sha256_ctx, (BYTE *)host_seed, 32);
    cuda_sha256_update(&sha256_ctx, (BYTE *)(&idx), 8);
    cuda_sha256_final(&sha256_ctx, (BYTE *)seed);

    unsigned long long start_clock = clock64();

    for (uint64_t iter = 0; iter < uint64_t(1000) * 1000 * 1000 * 1000; iter++)
    {
        if (iter % 100 == 0) {
            if (atomicMax(&kp_done, 0) == 1) {
                atomicAdd(&kp_count, iter);
                return;
            }
            if (clock64() - start_clock >= max_cycles) {
                atomicAdd(&kp_count, iter);
                return;
            }
        }

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
